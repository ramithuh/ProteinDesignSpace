// Pure-TS 3D B-factor painting (PDB & mmCIF), mutation ranking, and CSV export helpers
// for MutationMatrixArtifact & ResidueTrackArtifact visualizers.

import type { MutationMatrixArtifact, ResidueTrackArtifact } from '$lib/engine/types';

export interface PaintBfactorsOptions {
  /** Multiplicative scale applied to each per-residue value before clamping (default `1`). */
  scale?: number;
  /** Optional lower clamp applied after scaling. */
  clampMin?: number;
  /** Optional upper clamp applied after scaling. */
  clampMax?: number;
}

export interface RankedMutation {
  rank: number;
  positionIndex: number;
  residueId: string;
  chainId?: string;
  wildType: string;
  mutant: string;
  /** Human-readable mutation notation (e.g. `A42G` or `H:G100AD` for multi-chain). */
  label: string;
  score: number;
  mutantSequence: string;
}

const NUCLEIC_RESNAMES = new Set([
  'DA', 'DC', 'DG', 'DT', 'DU', 'DI',
  'A', 'C', 'G', 'U', 'T', 'I',
  'ADE', 'CYT', 'GUA', 'THY', 'URA',
]);

function padLeft(s: string, width: number): string {
  return s.length >= width ? s : ' '.repeat(width - s.length) + s;
}

function transformBfactorValue(raw: number, opts?: PaintBfactorsOptions): number | null {
  if (!Number.isFinite(raw)) return null;
  const scale = opts?.scale !== undefined && Number.isFinite(opts.scale) ? opts.scale : 1;
  let v = raw * scale;
  if (!Number.isFinite(v)) return null;
  if (opts?.clampMin !== undefined && Number.isFinite(opts.clampMin)) {
    v = Math.max(opts.clampMin, v);
  }
  if (opts?.clampMax !== undefined && Number.isFinite(opts.clampMax)) {
    v = Math.min(opts.clampMax, v);
  }
  return v;
}

/** Fixed-width PDB B-factor field (`cols 60..66`, `%6.2f`). */
function formatPdbBfactor6(v: number): string {
  const clamped = Math.max(-99.99, Math.min(999.99, v));
  const fixed = clamped.toFixed(2);
  // Avoid '-0.00'
  const normalized = fixed === '-0.00' ? '0.00' : fixed;
  return padLeft(normalized, 6);
}

function formatCifBfactor(v: number): string {
  const fixed = v.toFixed(2);
  return fixed === '-0.00' ? '0.00' : fixed;
}

interface CifTokenSpan {
  value: string;
  start: number;
  end: number;
}

/** Tokenize a single mmCIF data line while tracking character spans and respecting quotes. */
function tokenizeCifLineWithSpans(line: string): CifTokenSpan[] {
  const toks: CifTokenSpan[] = [];
  const n = line.length;
  let i = 0;
  while (i < n) {
    while (i < n && (line[i] === ' ' || line[i] === '\t')) i++;
    if (i >= n) break;
    const ch = line[i];
    if (ch === "'" || ch === '"') {
      const quote = ch;
      const tokStart = i;
      i++;
      const valStart = i;
      while (i < n) {
        if (line[i] === quote && (i + 1 === n || line[i + 1] === ' ' || line[i + 1] === '\t')) {
          break;
        }
        i++;
      }
      const value = line.slice(valStart, i);
      if (i < n && line[i] === quote) i++;
      toks.push({ value, start: tokStart, end: i });
    } else {
      const tokStart = i;
      while (i < n && line[i] !== ' ' && line[i] !== '\t') i++;
      toks.push({ value: line.slice(tokStart, i), start: tokStart, end: i });
    }
  }
  return toks;
}

function isPolymerPdbAtomLine(line: string): boolean {
  if (line.length < 27) return false;
  const isAtom = line.startsWith('ATOM  ');
  const isHet = line.startsWith('HETATM');
  if (!isAtom && !isHet) return false;
  const resName = line.slice(17, 20).trim().toUpperCase();
  if (!resName || NUCLEIC_RESNAMES.has(resName)) return false;
  if (isHet) return resName === 'MSE';
  return true;
}

function pdbResidueKey(line: string): string {
  const chain = line[21] ?? ' ';
  const resSeq = line.slice(22, 26).trim();
  const insCode = (line[26] ?? ' ').trim();
  return `${chain}:${resSeq}:${insCode}`;
}

function paintPdbBfactors(
  pdbText: string,
  perResidueValues: Float32Array | number[],
  opts?: PaintBfactorsOptions,
): string {
  const hasCrlf = pdbText.includes('\r\n');
  const rawLines = pdbText.split('\n');
  const lines = rawLines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));

  // Pass 1: collect ordered polymer residue keys per MODEL (preferring residues with CA to stay aligned with backbone parsers)
  const modelCaKeys: Map<number, Map<string, number>> = new Map();
  const modelAllKeys: Map<number, Map<string, number>> = new Map();
  let modelIdx = 0;
  let seenAtomInModel = false;

  const ensureModelMaps = (m: number) => {
    if (!modelCaKeys.has(m)) modelCaKeys.set(m, new Map());
    if (!modelAllKeys.has(m)) modelAllKeys.set(m, new Map());
  };

  for (const line of lines) {
    if (line.startsWith('ENDMDL') && seenAtomInModel) {
      modelIdx++;
      seenAtomInModel = false;
      continue;
    }
    if (!isPolymerPdbAtomLine(line)) continue;
    seenAtomInModel = true;
    ensureModelMaps(modelIdx);
    const key = pdbResidueKey(line);
    const allMap = modelAllKeys.get(modelIdx)!;
    if (!allMap.has(key)) allMap.set(key, allMap.size);
    const atomName = line.slice(12, 16).trim();
    if (atomName === 'CA') {
      const caMap = modelCaKeys.get(modelIdx)!;
      if (!caMap.has(key)) caMap.set(key, caMap.size);
    }
  }

  // Pass 2: rewrite B-factor columns (60..66, 0-based) for matched polymer residues only
  const outLines: string[] = new Array(lines.length);
  modelIdx = 0;
  seenAtomInModel = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('ENDMDL') && seenAtomInModel) {
      modelIdx++;
      seenAtomInModel = false;
      outLines[i] = line;
      continue;
    }
    if (!isPolymerPdbAtomLine(line) || line.length < 54) {
      outLines[i] = line;
      continue;
    }
    seenAtomInModel = true;
    const caMap = modelCaKeys.get(modelIdx);
    const keyMap = caMap && caMap.size > 0 ? caMap : modelAllKeys.get(modelIdx);
    const resIdx = keyMap?.get(pdbResidueKey(line));
    if (resIdx === undefined || resIdx >= perResidueValues.length) {
      outLines[i] = line;
      continue;
    }
    const painted = transformBfactorValue(perResidueValues[resIdx], opts);
    if (painted === null) {
      outLines[i] = line;
      continue;
    }
    const bStr = formatPdbBfactor6(painted);
    if (line.length >= 66) {
      outLines[i] = line.slice(0, 60) + bStr + line.slice(66);
    } else if (line.length >= 60) {
      outLines[i] = line.slice(0, 60) + bStr;
    } else {
      const rawOcc = line.slice(54);
      const occField = rawOcc.trim().length > 0 ? padLeft(rawOcc.trim().slice(0, 6), 6) : '  1.00';
      outLines[i] = line.slice(0, 54) + occField + bStr;
    }
  }

  return outLines.join(hasCrlf ? '\r\n' : '\n');
}

function cifClean(v: string): string {
  return v === '.' || v === '?' ? '' : v;
}

function paintMmcifBfactors(
  cifText: string,
  perResidueValues: Float32Array | number[],
  opts?: PaintBfactorsOptions,
): string {
  const hasCrlf = cifText.includes('\r\n');
  const rawLines = cifText.split('\n');
  const lines = rawLines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));

  // Locate the _atom_site loop block(s) and rewrite B_iso_or_equiv in place
  const cols: string[] = [];
  let inAtomSiteHeader = false;
  let inAtomSiteRows = false;
  let lastHeaderLineIdx = -1;
  const atomSiteRowIndices: number[] = [];

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (trimmed.startsWith('_atom_site.')) {
      if (inAtomSiteRows) break; // only first _atom_site loop
      inAtomSiteHeader = true;
      cols.push(trimmed.slice('_atom_site.'.length).split(/\s+/)[0]);
      lastHeaderLineIdx = i;
      continue;
    }
    if (inAtomSiteHeader) {
      if (
        trimmed === '' ||
        trimmed.startsWith('#') ||
        trimmed.startsWith('loop_') ||
        trimmed.startsWith('_') ||
        trimmed.startsWith('data_')
      ) {
        if (inAtomSiteRows) break;
        continue;
      }
      inAtomSiteRows = true;
      atomSiteRowIndices.push(i);
    }
  }

  if (cols.length === 0 || atomSiteRowIndices.length === 0) {
    return cifText;
  }

  const iGroup = cols.indexOf('group_PDB');
  const iModel = cols.indexOf('pdbx_PDB_model_num');
  const iLabelAtom = cols.indexOf('label_atom_id');
  const iAuthAtom = cols.indexOf('auth_atom_id');
  const iAuthAsym = cols.indexOf('auth_asym_id');
  const iLabelAsym = cols.indexOf('label_asym_id');
  const iAuthSeq = cols.indexOf('auth_seq_id');
  const iLabelSeq = cols.indexOf('label_seq_id');
  const iIns = cols.indexOf('pdbx_PDB_ins_code');
  const iLabelComp = cols.indexOf('label_comp_id');
  const iAuthComp = cols.indexOf('auth_comp_id');
  const iBiso = cols.indexOf('B_iso_or_equiv');

  const getTok = (toks: CifTokenSpan[], idx: number): string =>
    idx >= 0 && idx < toks.length ? cifClean(toks[idx].value) : '';

  interface ParsedRowMeta {
    lineIdx: number;
    toks: CifTokenSpan[];
    isPolymer: boolean;
    modelNum: string;
    resKey: string;
    atomName: string;
  }

  const parsedRows: ParsedRowMeta[] = [];
  const caKeysByModel = new Map<string, Map<string, number>>();
  const allKeysByModel = new Map<string, Map<string, number>>();

  const ensureModel = (m: string) => {
    if (!caKeysByModel.has(m)) caKeysByModel.set(m, new Map());
    if (!allKeysByModel.has(m)) allKeysByModel.set(m, new Map());
  };

  for (const lineIdx of atomSiteRowIndices) {
    const toks = tokenizeCifLineWithSpans(lines[lineIdx]);
    if (toks.length < cols.length) {
      parsedRows.push({
        lineIdx,
        toks,
        isPolymer: false,
        modelNum: '1',
        resKey: '',
        atomName: '',
      });
      continue;
    }
    const group = (getTok(toks, iGroup) || 'ATOM').toUpperCase();
    const comp = (getTok(toks, iLabelComp) || getTok(toks, iAuthComp) || '').toUpperCase();
    const isPolymer =
      Boolean(comp) &&
      !NUCLEIC_RESNAMES.has(comp) &&
      (group === 'ATOM' || (group === 'HETATM' && comp === 'MSE'));
    const modelNum = getTok(toks, iModel) || '1';
    const chain = getTok(toks, iAuthAsym) || getTok(toks, iLabelAsym) || 'A';
    const seqId = getTok(toks, iAuthSeq) || getTok(toks, iLabelSeq);
    const insCode = getTok(toks, iIns);
    const atomName = getTok(toks, iLabelAtom) || getTok(toks, iAuthAtom);
    const resKey = `${chain}:${seqId}:${insCode}`;

    if (isPolymer) {
      ensureModel(modelNum);
      const allMap = allKeysByModel.get(modelNum)!;
      if (!allMap.has(resKey)) allMap.set(resKey, allMap.size);
      if (atomName === 'CA') {
        const caMap = caKeysByModel.get(modelNum)!;
        if (!caMap.has(resKey)) caMap.set(resKey, caMap.size);
      }
    }

    parsedRows.push({ lineIdx, toks, isPolymer, modelNum, resKey, atomName });
  }

  const outLines = lines.slice();
  const needsNewBisoColumn = iBiso < 0;

  for (const row of parsedRows) {
    if (row.toks.length < cols.length) continue;
    let painted: number | null = null;
    if (row.isPolymer) {
      const caMap = caKeysByModel.get(row.modelNum);
      const keyMap = caMap && caMap.size > 0 ? caMap : allKeysByModel.get(row.modelNum);
      const resIdx = keyMap?.get(row.resKey);
      if (resIdx !== undefined && resIdx < perResidueValues.length) {
        painted = transformBfactorValue(perResidueValues[resIdx], opts);
      }
    }

    if (needsNewBisoColumn) {
      const bVal = painted !== null ? formatCifBfactor(painted) : '0.00';
      outLines[row.lineIdx] = `${lines[row.lineIdx]} ${bVal}`;
    } else if (painted !== null) {
      const bTok = row.toks[iBiso];
      const bVal = formatCifBfactor(painted);
      outLines[row.lineIdx] =
        lines[row.lineIdx].slice(0, bTok.start) + bVal + lines[row.lineIdx].slice(bTok.end);
    }
  }

  if (needsNewBisoColumn && lastHeaderLineIdx >= 0) {
    outLines.splice(lastHeaderLineIdx + 1, 0, '_atom_site.B_iso_or_equiv');
  }

  return outLines.join(hasCrlf ? '\r\n' : '\n');
}

/**
 * Rewrite per-residue B-factor values in PDB (`cols 60..66`, fixed-width `%6.2f`) or
 * mmCIF (`_atom_site.B_iso_or_equiv`) per polymer residue in residue order, while
 * preserving ligands, ions, waters, nucleic acids, insertion codes, and multi-char chains.
 */
export function paintStructureBfactors(
  structureText: string,
  perResidueValues: Float32Array | number[],
  opts?: PaintBfactorsOptions,
): string {
  if (!structureText || !perResidueValues || perResidueValues.length === 0) {
    return structureText;
  }
  const trimmed = structureText.trimStart();
  const isCif =
    trimmed.startsWith('data_') || /(?:^|\r?\n)\s*_atom_site\./.test(structureText);
  if (isCif) {
    return paintMmcifBfactors(structureText, perResidueValues, opts);
  }
  return paintPdbBfactors(structureText, perResidueValues, opts);
}

/** Parse an insertion-code-safe residue label (e.g. `A42`, `H100A`, `AA:12B`, `42`) into chain & position. */
export function parseResidueLabel(residueId: string): { chain: string; pos: string } {
  const s = (residueId ?? '').trim();
  if (!s) return { chain: '', pos: '' };
  const colon = /^([A-Za-z0-9_]+):(-?\d+[A-Za-z]?)$/.exec(s);
  if (colon) return { chain: colon[1], pos: colon[2] };
  const prefixed = /^([A-Za-z]+)(-?\d+[A-Za-z]?)$/.exec(s);
  if (prefixed) return { chain: prefixed[1], pos: prefixed[2] };
  const bare = /^(-?\d+[A-Za-z]?)$/.exec(s);
  if (bare) return { chain: '', pos: bare[1] };
  return { chain: '', pos: s };
}

/** Format a mutation label (`A42G` for single-chain, `H:G100AD` when multiple chains exist). */
export function formatMutationLabel(
  wildType: string,
  residueId: string,
  mutant: string,
  multiChain: boolean,
  fallbackPosition1Based?: number,
): string {
  const wt = (wildType || 'X').toUpperCase();
  const mut = (mutant || 'X').toUpperCase();
  const parsed = parseResidueLabel(residueId);
  const pos = parsed.pos || (fallbackPosition1Based !== undefined ? String(fallbackPosition1Based) : '');
  const core = `${wt}${pos}${mut}`;
  return multiChain && parsed.chain ? `${parsed.chain}:${core}` : core;
}

/** Detect whether `residueIds` spans more than one distinct chain prefix. */
export function hasMultipleResidueChains(residueIds: readonly string[]): boolean {
  let firstChain = '';
  for (const id of residueIds) {
    const { chain } = parseResidueLabel(id);
    if (!chain) continue;
    if (!firstChain) firstChain = chain;
    else if (chain !== firstChain) return true;
  }
  return false;
}

/**
 * Rank the top beneficial/stabilizing point mutations from a `MutationMatrixArtifact`.
 * Respects `matrix.direction` (`'lower-is-better'` ascending vs `'higher-is-better'` descending),
 * skips wild-type self-mutations and non-finite (`NaN` / `±Infinity`) scores.
 */
export interface RankMutationsOptions {
  sort?: 'best' | 'worst';
  query?: string;
  limit?: number;
}

export interface RankedMutationsResult {
  items: RankedMutation[];
  totalMatched: number;
  totalAll: number;
}

function matchesMutationToken(
  token: string,
  c: { residueId: string; wildType: string; mutant: string; label: string; positionIndex: number },
): boolean {
  const t = token.trim().toUpperCase();
  if (!t) return true;
  const labelU = c.label.toUpperCase();
  const resIdU = c.residueId.toUpperCase();
  const parsed = parseResidueLabel(c.residueId);
  const numStr = parsed.pos || String(c.positionIndex + 1);
  const wtNum = `${c.wildType}${numStr}`;
  const wtNumMut = `${c.wildType}${numStr}${c.mutant}`;

  if (/^\d+$/.test(t)) {
    return numStr === t;
  }
  if (labelU === t || resIdU === t || wtNum === t || wtNumMut === t) {
    return true;
  }
  // Support chain-prefixed shorthand like "A12" or "A:12" or "G12" prefix
  if (t.length >= 2 && (labelU.startsWith(t) || resIdU === t.replace(':', ''))) {
    return true;
  }
  return false;
}

export function rankMutations(
  matrix: MutationMatrixArtifact,
  opts: RankMutationsOptions = {},
): RankedMutationsResult {
  const sort = opts.sort === 'worst' ? 'worst' : 'best';
  const maxItems = Math.max(0, Math.floor(opts.limit ?? 10));
  const queryTokens = (opts.query ?? '')
    .split(/[,\s;]+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const residues = (matrix.residues ?? '').toUpperCase();
  const L = residues.length;
  const alphabet = (matrix.alphabet || 'ACDEFGHIKLMNPQRSTVWY').toUpperCase();
  const K = alphabet.length;
  const residueIds = matrix.residueIds ?? [];
  const multiChain = hasMultipleResidueChains(residueIds);
  const lowerBetter = matrix.direction === 'lower-is-better';

  interface Candidate {
    positionIndex: number;
    residueId: string;
    chainId?: string;
    wildType: string;
    mutant: string;
    label: string;
    score: number;
  }

  const candidates: Candidate[] = [];

  if (L > 0 && K > 0 && matrix.values && matrix.values.length > 0) {
    for (let i = 0; i < L; i++) {
      const wt = residues[i];
      const resId = residueIds[i] ?? String(i + 1);
      const parsed = parseResidueLabel(resId);
      const rowOff = i * K;
      for (let j = 0; j < K; j++) {
        const idx = rowOff + j;
        if (idx >= matrix.values.length) break;
        const mut = alphabet[j];
        if (!mut || mut === wt) continue;
        const score = matrix.values[idx];
        if (!Number.isFinite(score)) continue;
        candidates.push({
          positionIndex: i,
          residueId: resId,
          ...(parsed.chain ? { chainId: parsed.chain } : {}),
          wildType: wt,
          mutant: mut,
          label: formatMutationLabel(wt, resId, mut, multiChain, i + 1),
          score,
        });
      }
    }
  } else if (matrix.targetedMutations && matrix.targetedMutations.length > 0) {
    const idToIdx = new Map<string, number>();
    for (let i = 0; i < residueIds.length; i++) {
      idToIdx.set(residueIds[i], i);
    }
    for (const tm of matrix.targetedMutations) {
      const wt = (tm.wildType || '').toUpperCase();
      const mut = (tm.mutant || '').toUpperCase();
      if (!mut || mut === wt || !Number.isFinite(tm.score)) continue;
      const resId = tm.residueId ?? '';
      const posIdx = resId && idToIdx.has(resId) ? idToIdx.get(resId)! : -1;
      candidates.push({
        positionIndex: posIdx,
        residueId: resId || tm.mutation,
        ...(tm.chainId ? { chainId: tm.chainId } : {}),
        wildType: wt,
        mutant: mut,
        label: tm.mutation || formatMutationLabel(wt, resId, mut, multiChain),
        score: tm.score,
      });
    }
  }

  // Sort in the requested direction so rank #1 is best (or most deleterious for 'worst')
  const ascending = sort === 'best' ? lowerBetter : !lowerBetter;
  candidates.sort((a, b) => {
    const diff = ascending ? a.score - b.score : b.score - a.score;
    if (diff !== 0) return diff;
    if (a.positionIndex !== b.positionIndex) return a.positionIndex - b.positionIndex;
    return a.mutant.localeCompare(b.mutant);
  });

  const totalAll = candidates.length;
  const ranked: Array<Candidate & { rank: number }> = [];
  for (let idx = 0; idx < candidates.length; idx++) {
    const c = candidates[idx];
    if (queryTokens.length > 0 && !queryTokens.some((tok) => matchesMutationToken(tok, c))) {
      continue;
    }
    ranked.push({ ...c, rank: idx + 1 });
  }

  const totalMatched = ranked.length;
  const sliced = maxItems === 0 ? [] : ranked.slice(0, maxItems);
  const items = sliced.map((c) => {
    const mutantSequence =
      c.positionIndex >= 0 && c.positionIndex < L
        ? residues.slice(0, c.positionIndex) + c.mutant + residues.slice(c.positionIndex + 1)
        : residues;
    return {
      ...c,
      mutantSequence,
    };
  });

  return { items, totalMatched, totalAll };
}

export function rankTopMutations(
  matrix: MutationMatrixArtifact,
  limit = 10,
): RankedMutation[] {
  return rankMutations(matrix, { sort: 'best', limit }).items;
}

function escapeCsvCell(raw: string): string {
  const isNumeric = /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(raw);
  const safe = !isNumeric && /^[=+\-@\t\r]/.test(raw) ? `'${raw}` : raw;
  if (safe.includes(',') || safe.includes('"') || safe.includes('\n') || safe.includes('\r')) {
    return `"${safe.replace(/"/g, '""')}"`;
  }
  return safe;
}

/** Format a `MutationMatrixArtifact` as RFC-4180 CSV (`position,residue_id,wild_type,A,C,...,Y`). */
export function formatMutationMatrixCsv(matrix: MutationMatrixArtifact): string {
  const residues = (matrix.residues ?? '').toUpperCase();
  const L = residues.length;
  const alphabet = (matrix.alphabet || 'ACDEFGHIKLMNPQRSTVWY').toUpperCase();
  const K = alphabet.length;
  const hasMean = Boolean(matrix.meanPerPosition && matrix.meanPerPosition.length >= L);

  const headerCols = ['position', 'residue_id', 'wild_type', ...alphabet.split('')];
  if (hasMean) headerCols.push('mean');

  const lines: string[] = [headerCols.join(',')];
  for (let i = 0; i < L; i++) {
    const resId = matrix.residueIds?.[i] ?? String(i + 1);
    const wt = residues[i] ?? 'X';
    const row: string[] = [String(i + 1), escapeCsvCell(resId), wt];
    const rowOff = i * K;
    for (let j = 0; j < K; j++) {
      const idx = rowOff + j;
      const v = idx < matrix.values.length ? matrix.values[idx] : Number.NaN;
      row.push(Number.isFinite(v) ? v.toFixed(4) : '');
    }
    if (hasMean) {
      const m = matrix.meanPerPosition![i];
      row.push(Number.isFinite(m) ? m.toFixed(4) : '');
    }
    lines.push(row.join(','));
  }
  return lines.join('\n') + '\n';
}

/** Format a `ResidueTrackArtifact` as RFC-4180 CSV (`position,residue_id,residue,value[,raw_logit]`). */
export function formatResidueTrackCsv(track: ResidueTrackArtifact): string {
  const L = Math.max(track.values?.length ?? 0, track.residues?.length ?? 0);
  const hasLogits = Boolean(track.rawLogits && track.rawLogits.length > 0);
  const header = hasLogits
    ? 'position,residue_id,residue,value,raw_logit'
    : 'position,residue_id,residue,value';
  const lines: string[] = [header];

  for (let i = 0; i < L; i++) {
    const resId = track.residueIds?.[i] ?? String(i + 1);
    const aa = (track.residues?.[i] ?? 'X').toUpperCase();
    const v = i < (track.values?.length ?? 0) ? track.values[i] : Number.NaN;
    const row = [
      String(i + 1),
      escapeCsvCell(resId),
      aa,
      Number.isFinite(v) ? v.toFixed(6) : '',
    ];
    if (hasLogits) {
      const logit = i < track.rawLogits!.length ? track.rawLogits![i] : Number.NaN;
      row.push(Number.isFinite(logit) ? logit.toFixed(6) : '');
    }
    lines.push(row.join(','));
  }
  return lines.join('\n') + '\n';
}
