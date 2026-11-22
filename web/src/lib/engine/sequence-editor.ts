export type AaChemGroup =
  | 'hydrophobic'
  | 'aromatic'
  | 'polar'
  | 'positive'
  | 'negative'
  | 'special';

export interface AminoAcidMeta {
  code: string;
  three: string;
  name: string;
  group: AaChemGroup;
}

export const STANDARD_AAS: AminoAcidMeta[] = [
  { code: 'A', three: 'Ala', name: 'Alanine', group: 'hydrophobic' },
  { code: 'C', three: 'Cys', name: 'Cysteine', group: 'special' },
  { code: 'D', three: 'Asp', name: 'Aspartate', group: 'negative' },
  { code: 'E', three: 'Glu', name: 'Glutamate', group: 'negative' },
  { code: 'F', three: 'Phe', name: 'Phenylalanine', group: 'aromatic' },
  { code: 'G', three: 'Gly', name: 'Glycine', group: 'special' },
  { code: 'H', three: 'His', name: 'Histidine', group: 'positive' },
  { code: 'I', three: 'Ile', name: 'Isoleucine', group: 'hydrophobic' },
  { code: 'K', three: 'Lys', name: 'Lysine', group: 'positive' },
  { code: 'L', three: 'Leu', name: 'Leucine', group: 'hydrophobic' },
  { code: 'M', three: 'Met', name: 'Methionine', group: 'hydrophobic' },
  { code: 'N', three: 'Asn', name: 'Asparagine', group: 'polar' },
  { code: 'P', three: 'Pro', name: 'Proline', group: 'special' },
  { code: 'Q', three: 'Gln', name: 'Glutamine', group: 'polar' },
  { code: 'R', three: 'Arg', name: 'Arginine', group: 'positive' },
  { code: 'S', three: 'Ser', name: 'Serine', group: 'polar' },
  { code: 'T', three: 'Thr', name: 'Threonine', group: 'polar' },
  { code: 'V', three: 'Val', name: 'Valine', group: 'hydrophobic' },
  { code: 'W', three: 'Trp', name: 'Tryptophan', group: 'aromatic' },
  { code: 'Y', three: 'Tyr', name: 'Tyrosine', group: 'aromatic' },
];

export const AA_BY_CODE: Record<string, AminoAcidMeta> = Object.fromEntries(
  STANDARD_AAS.map((m) => [m.code, m]),
);

export const STANDARD_AA_SET = new Set(STANDARD_AAS.map((m) => m.code));

export interface ParsedResidueCell {
  /** 1-indexed position across the entry's residues. */
  pos: number;
  /** 0-indexed chain within the entry (incremented on ':' or '/'). */
  chainIdx: number;
  /** 1-indexed position within the current chain. */
  chainPos: number;
  /** Uppercase single-letter amino acid code (or raw letter if non-canonical). */
  aa: string;
  /** Exact UTF-16 character index in the raw input string. */
  charIndex: number;
  /** True if preceded by a chain break (':' or '/'). */
  startsNewChain: boolean;
}

export interface ParsedSequenceEntry {
  entryIdx: number;
  header: string;
  residues: ParsedResidueCell[];
}

export function parseSequenceEntries(raw: string): ParsedSequenceEntry[] {
  const entries: ParsedSequenceEntry[] = [];
  let current: ParsedSequenceEntry | null = null;
  let pos = 0;
  let chainIdx = 0;
  let chainPos = 0;
  let pendingChainBreak = false;

  const ensureEntry = (header = '') => {
    if (!current) {
      current = { entryIdx: entries.length, header, residues: [] };
      entries.push(current);
      pos = 0;
      chainIdx = 0;
      chainPos = 0;
      pendingChainBreak = false;
    }
    return current;
  };

  let i = 0;
  const n = raw.length;
  while (i < n) {
    const lineStart = i;
    let lineEnd = raw.indexOf('\n', i);
    if (lineEnd === -1) lineEnd = n;
    const line = raw.slice(lineStart, lineEnd);
    const trimmed = line.trim();

    if (trimmed.startsWith('>')) {
      current = {
        entryIdx: entries.length,
        header: trimmed.slice(1).trim(),
        residues: [],
      };
      entries.push(current);
      pos = 0;
      chainIdx = 0;
      chainPos = 0;
      pendingChainBreak = false;
    } else if (trimmed.length > 0) {
      const entry = ensureEntry();
      for (let k = 0; k < line.length; k++) {
        const ch = line[k];
        if (ch === ' ' || ch === '\t' || ch === '\r') continue;
        if (ch === ':' || ch === '/') {
          if (entry.residues.length > 0 && !pendingChainBreak) {
            chainIdx += 1;
            chainPos = 0;
            pendingChainBreak = true;
          }
          continue;
        }
        if (/[A-Za-z*]/.test(ch)) {
          pos += 1;
          chainPos += 1;
          entry.residues.push({
            pos,
            chainIdx,
            chainPos,
            aa: ch.toUpperCase(),
            charIndex: lineStart + k,
            startsNewChain: pendingChainBreak,
          });
          pendingChainBreak = false;
        }
      }
    }

    i = lineEnd + 1;
  }

  return entries;
}

export function mutateResidueInRaw(
  raw: string,
  entryIdx: number,
  pos1: number,
  nextAa: string,
): string | null {
  const upper = nextAa.trim().toUpperCase();
  if (!STANDARD_AA_SET.has(upper)) return null;
  const entries = parseSequenceEntries(raw);
  const entry = entries[entryIdx];
  if (!entry) return null;
  const cell = entry.residues[pos1 - 1];
  if (!cell) return null;
  return raw.slice(0, cell.charIndex) + upper + raw.slice(cell.charIndex + 1);
}

export type MutationCommandResult =
  | { kind: 'jump'; pos: number }
  | { kind: 'mutate'; pos: number; targetAa: string; expectedWt?: string }
  | { kind: 'error'; message: string };

/**
 * Parse shorthand biologist mutation or position jump commands:
 * - "45" -> jump to pos 45
 * - "Y45L" / "Y45->L" / "Y45→L" -> verify pos 45 is Y, mutate to L
 * - "45L" / "45->L" / "45→L" / "45 L" -> mutate pos 45 to L
 */
export function parseMutationCommand(input: string): MutationCommandResult {
  const s = input.trim().replace(/\s+/g, '');
  if (!s) return { kind: 'error', message: 'Enter a position (e.g. 45) or mutation (e.g. Y45L).' };

  const jumpOnly = /^(\d+)$/.exec(s);
  if (jumpOnly) {
    const pos = Number(jumpOnly[1]);
    if (pos < 1) return { kind: 'error', message: 'Position must be ≥ 1.' };
    return { kind: 'jump', pos };
  }

  const fullMut = /^([A-Za-z])(\d+)(?:->|→|to)?([A-Za-z])$/i.exec(s);
  if (fullMut) {
    const expectedWt = fullMut[1].toUpperCase();
    const pos = Number(fullMut[2]);
    const targetAa = fullMut[3].toUpperCase();
    if (!STANDARD_AA_SET.has(expectedWt)) {
      return { kind: 'error', message: `Unknown residue "${expectedWt}".` };
    }
    if (!STANDARD_AA_SET.has(targetAa)) {
      return { kind: 'error', message: `Unknown target amino acid "${targetAa}".` };
    }
    if (pos < 1) return { kind: 'error', message: 'Position must be ≥ 1.' };
    return { kind: 'mutate', expectedWt, pos, targetAa };
  }

  const posMut = /^(\d+)(?:->|→|:)?([A-Za-z])$/i.exec(s);
  if (posMut) {
    const pos = Number(posMut[1]);
    const targetAa = posMut[2].toUpperCase();
    if (!STANDARD_AA_SET.has(targetAa)) {
      return { kind: 'error', message: `Unknown target amino acid "${targetAa}".` };
    }
    if (pos < 1) return { kind: 'error', message: 'Position must be ≥ 1.' };
    return { kind: 'mutate', pos, targetAa };
  }

  return { kind: 'error', message: 'Use format like Y45L, 45L, or 45.' };
}

export interface PointMutationDiff {
  pos: number;
  from: string;
  to: string;
  label: string;
}

export function diffEntryResidues(
  baselineRaw: string,
  currentRaw: string,
  entryIdx = 0,
): PointMutationDiff[] {
  const baseEntry = parseSequenceEntries(baselineRaw)[entryIdx];
  const curEntry = parseSequenceEntries(currentRaw)[entryIdx];
  if (!baseEntry || !curEntry) return [];
  if (baseEntry.residues.length !== curEntry.residues.length) return [];
  const diffs: PointMutationDiff[] = [];
  for (let i = 0; i < curEntry.residues.length; i++) {
    const a = baseEntry.residues[i].aa;
    const b = curEntry.residues[i].aa;
    if (a !== b) {
      const pos = i + 1;
      diffs.push({ pos, from: a, to: b, label: `${a}${pos}${b}` });
    }
  }
  return diffs;
}

export interface CursorResidueInfo {
  entryIdx: number;
  pos: number;
  aa: string;
  total: number;
}

export function cursorToResiduePosition(
  raw: string,
  cursorOffset: number,
): CursorResidueInfo | null {
  const entries = parseSequenceEntries(raw);
  let fallback: CursorResidueInfo | null = null;
  for (const entry of entries) {
    for (const cell of entry.residues) {
      if (cell.charIndex === cursorOffset) {
        return {
          entryIdx: entry.entryIdx,
          pos: cell.pos,
          aa: cell.aa,
          total: entry.residues.length,
        };
      }
      if (cell.charIndex + 1 === cursorOffset) {
        fallback = {
          entryIdx: entry.entryIdx,
          pos: cell.pos,
          aa: cell.aa,
          total: entry.residues.length,
        };
      }
    }
  }
  return fallback;
}
