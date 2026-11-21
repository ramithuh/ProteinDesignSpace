import type { DevelopabilityLiability, DevelopabilitySummary, PreviewResidue } from './studio-types';

export interface ParsedResidueToken {
  chain: string;
  seqId: number;
  insCode: string;
  raw: string;
}

const SINGLE_TOKEN_RE = /^([A-Za-z_])?:?(-?\d+)([A-Za-z])?$/;
const RANGE_TOKEN_RE = /^([A-Za-z_])?:?(-?\d+)-([A-Za-z_])?:?(-?\d+)$/;
const CHAIN_ONLY_RE = /^([A-Za-z_])$/;

export function formatResidueToken(chain: string, seqId: number, insCode = ''): string {
  const ch = chain.trim().toUpperCase() || 'A';
  const ins = insCode.trim();
  return `${ch}${seqId}${ins}`;
}

/**
 * Parse a comma/whitespace-separated residue list into expanded tokens + whole-chain tokens.
 * Preserves negative residue numbers (`A-2`, `A-5--2`) and insertion codes (`H100A`).
 */
export function expandResidueTokenList(
  spec: unknown,
  defaultChain = 'A',
): {
  wholeChains: string[];
  tokens: ParsedResidueToken[];
  passthrough: string[];
} {
  const text = String(spec ?? '').trim();
  const wholeChains: string[] = [];
  const tokens: ParsedResidueToken[] = [];
  const passthrough: string[] = [];
  if (!text) return { wholeChains, tokens, passthrough };

  const seenChains = new Set<string>();
  const seenTokens = new Set<string>();

  for (const raw of text.split(/[,\s]+/).filter(Boolean)) {
    const chOnly = CHAIN_ONLY_RE.exec(raw);
    if (chOnly) {
      const ch = chOnly[1].toUpperCase();
      if (!seenChains.has(ch)) {
        seenChains.add(ch);
        wholeChains.push(ch);
      }
      continue;
    }

    const range = RANGE_TOKEN_RE.exec(raw);
    if (range) {
      const startCh = range[1] ? range[1].toUpperCase() : null;
      const endCh = range[3] ? range[3].toUpperCase() : null;
      if (startCh && endCh && startCh !== endCh) {
        passthrough.push(raw);
        continue;
      }
      const ch = startCh ?? endCh ?? defaultChain;
      const a = Number(range[2]);
      const b = Number(range[4]);
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      if (hi - lo > 5000) {
        passthrough.push(raw);
        continue;
      }
      for (let pos = lo; pos <= hi; pos++) {
        const key = `${ch}:${pos}:`;
        if (!seenTokens.has(key)) {
          seenTokens.add(key);
          tokens.push({ chain: ch, seqId: pos, insCode: '', raw: formatResidueToken(ch, pos) });
        }
      }
      continue;
    }

    const single = SINGLE_TOKEN_RE.exec(raw);
    if (single) {
      const ch = single[1] ? single[1].toUpperCase() : defaultChain;
      const seqId = Number(single[2]);
      const insCode = single[3] ?? '';
      const key = `${ch}:${seqId}:${insCode}`;
      if (!seenTokens.has(key)) {
        seenTokens.add(key);
        tokens.push({ chain: ch, seqId, insCode, raw: formatResidueToken(ch, seqId, insCode) });
      }
      continue;
    }

    passthrough.push(raw);
  }

  return { wholeChains, tokens, passthrough };
}

/**
 * Compact a set of residue tokens into minimal human-readable ranges (`A10-15, H100A`).
 * Contiguous runs with empty `insCode` collapse into `A10-15`; insertion codes remain explicit.
 */
export function compactResidueTokens(
  tokens: ReadonlyArray<{ chain: string; seqId: number; insCode?: string }>,
  opts: { wholeChains?: readonly string[]; passthrough?: readonly string[] } = {},
): string {
  const parts: string[] = [];
  const wholeSet = new Set((opts.wholeChains ?? []).map((c) => c.trim().toUpperCase()).filter(Boolean));
  for (const ch of wholeSet) parts.push(ch);

  // Dedupe & group by chain in stable order
  const byChain = new Map<string, Array<{ seqId: number; insCode: string }>>();
  const seen = new Set<string>();
  for (const t of tokens) {
    const ch = (t.chain || 'A').trim().toUpperCase();
    if (wholeSet.has(ch)) continue;
    const ins = (t.insCode ?? '').trim();
    const key = `${ch}:${t.seqId}:${ins}`;
    if (seen.has(key)) continue;
    seen.add(key);
    let arr = byChain.get(ch);
    if (!arr) {
      arr = [];
      byChain.set(ch, arr);
    }
    arr.push({ seqId: t.seqId, insCode: ins });
  }

  for (const [ch, list] of byChain.entries()) {
    list.sort((a, b) => a.seqId - b.seqId || a.insCode.localeCompare(b.insCode));
    let i = 0;
    while (i < list.length) {
      const cur = list[i];
      if (cur.insCode) {
        parts.push(formatResidueToken(ch, cur.seqId, cur.insCode));
        i++;
        continue;
      }
      let j = i;
      while (
        j + 1 < list.length &&
        !list[j + 1].insCode &&
        list[j + 1].seqId === list[j].seqId + 1
      ) {
        j++;
      }
      const runLen = j - i + 1;
      if (runLen >= 3) {
        parts.push(`${ch}${list[i].seqId}-${list[j].seqId}`);
      } else {
        for (let k = i; k <= j; k++) {
          parts.push(formatResidueToken(ch, list[k].seqId));
        }
      }
      i = j + 1;
    }
  }

  for (const extra of opts.passthrough ?? []) {
    if (extra.trim()) parts.push(extra.trim());
  }

  return parts.join(', ');
}

/**
 * Toggle, add, or remove one or more residues in an existing residue-list parameter string.
 */
export function toggleResidueTokens(
  existingSpec: unknown,
  selected: ReadonlyArray<Pick<PreviewResidue, 'authChain' | 'authSeqId' | 'insCode'>>,
  mode: 'toggle' | 'add' | 'remove' = 'toggle',
  defaultChain = 'A',
): string {
  const parsed = expandResidueTokenList(existingSpec, defaultChain);
  const tokenMap = new Map<string, { chain: string; seqId: number; insCode: string }>();
  for (const t of parsed.tokens) {
    tokenMap.set(`${t.chain}:${t.seqId}:${t.insCode}`, {
      chain: t.chain,
      seqId: t.seqId,
      insCode: t.insCode,
    });
  }

  const normalizedSel = selected.map((s) => ({
    chain: (s.authChain || defaultChain).trim().toUpperCase(),
    seqId: s.authSeqId,
    insCode: (s.insCode ?? '').trim(),
  }));

  let effectiveMode: 'add' | 'remove';
  if (mode === 'toggle') {
    const allPresent =
      normalizedSel.length > 0 &&
      normalizedSel.every((s) => tokenMap.has(`${s.chain}:${s.seqId}:${s.insCode}`));
    effectiveMode = allPresent ? 'remove' : 'add';
  } else {
    effectiveMode = mode;
  }

  for (const s of normalizedSel) {
    const key = `${s.chain}:${s.seqId}:${s.insCode}`;
    if (effectiveMode === 'add') {
      tokenMap.set(key, s);
    } else {
      tokenMap.delete(key);
    }
  }

  return compactResidueTokens([...tokenMap.values()], {
    wholeChains: parsed.wholeChains,
    passthrough: parsed.passthrough,
  });
}

// ---------------------------------------------------------------------------
// Wet-lab biophysics & developability calculations (pure TS, zero deps)
// ---------------------------------------------------------------------------

/** Average residue monoisotopic-adjusted molecular weights (Da, minus H2O). */
const AA_RESIDUE_MW: Record<string, number> = {
  A: 71.0788,
  R: 156.1875,
  N: 114.1038,
  D: 115.0886,
  C: 103.1388,
  E: 129.1155,
  Q: 128.1307,
  G: 57.0519,
  H: 137.1411,
  I: 113.1594,
  L: 113.1594,
  K: 128.1741,
  M: 131.1926,
  F: 147.1766,
  P: 97.1167,
  S: 87.0782,
  T: 101.1051,
  W: 186.2132,
  Y: 163.176,
  V: 99.1326,
};

const WATER_MW = 18.01528;
const HYDROPHOBIC_SET = new Set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y']);

/** Compute Henderson-Hasselbalch net charge at a given pH (EMBOSS/Bjellqvist pKa values). */
function netChargeAtPh(counts: Record<string, number>, pH: number): number {
  const pos =
    1 / (1 + Math.pow(10, pH - 8.6)) + // N-terminus
    (counts.K ?? 0) / (1 + Math.pow(10, pH - 10.8)) +
    (counts.R ?? 0) / (1 + Math.pow(10, pH - 12.5)) +
    (counts.H ?? 0) / (1 + Math.pow(10, pH - 6.5));
  const neg =
    1 / (1 + Math.pow(10, 3.6 - pH)) + // C-terminus
    (counts.D ?? 0) / (1 + Math.pow(10, 3.9 - pH)) +
    (counts.E ?? 0) / (1 + Math.pow(10, 4.1 - pH)) +
    (counts.C ?? 0) / (1 + Math.pow(10, 8.5 - pH)) +
    (counts.Y ?? 0) / (1 + Math.pow(10, 10.1 - pH));
  return pos - neg;
}

export function computeDevelopability(sequence: string): DevelopabilitySummary {
  const clean = sequence.toUpperCase().replace(/[^A-Z]/g, '');
  const n = clean.length;
  if (n === 0) {
    return {
      length: 0,
      mwKda: 0,
      pI: 7.0,
      netCharge74: 0,
      extCoeff280: 0,
      cysCount: 0,
      liabilities: [],
    };
  }

  const counts: Record<string, number> = {};
  let mwDa = WATER_MW;
  const cysIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    const aa = clean[i];
    counts[aa] = (counts[aa] ?? 0) + 1;
    mwDa += AA_RESIDUE_MW[aa] ?? 110.0;
    if (aa === 'C') cysIndices.push(i);
  }

  // Bisection for isoelectric point pI
  let lo = 0.0;
  let hi = 14.0;
  for (let iter = 0; iter < 30; iter++) {
    const mid = 0.5 * (lo + hi);
    if (netChargeAtPh(counts, mid) > 0) lo = mid;
    else hi = mid;
  }
  const pI = 0.5 * (lo + hi);
  const netCharge74 = netChargeAtPh(counts, 7.4);

  // Pace et al. (1995) reduced extinction coefficient at 280 nm (M^-1 cm^-1)
  const extCoeff280 = (counts.W ?? 0) * 5500 + (counts.Y ?? 0) * 1490 + Math.floor((counts.C ?? 0) / 2) * 125;

  const liabilities: DevelopabilityLiability[] = [];

  if (cysIndices.length > 0) {
    const odd = cysIndices.length % 2 === 1;
    liabilities.push({
      kind: 'cys',
      severity: odd ? 'warn' : 'info',
      label: odd ? `Cys: ${cysIndices.length} (odd/unpaired)` : `Cys: ${cysIndices.length}`,
      detail: odd
        ? 'Odd number of cysteines risks free-thiol dimerization or aggregation during expression.'
        : 'Even cysteine count — inspect 3D proximity if disulfides are intended.',
      indices: cysIndices,
    });
  }

  const nglycIndices: number[] = [];
  const deamIndices: number[] = [];
  const isomIndices: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    const a = clean[i];
    const b = clean[i + 1];
    if (i < n - 2 && a === 'N' && b !== 'P' && (clean[i + 2] === 'S' || clean[i + 2] === 'T')) {
      nglycIndices.push(i);
    }
    if (a === 'N' && b === 'G') {
      deamIndices.push(i);
    }
    if (a === 'D' && b === 'G') {
      isomIndices.push(i);
    }
  }

  if (nglycIndices.length > 0) {
    liabilities.push({
      kind: 'nglyc',
      severity: 'warn',
      label: `N-Glyc: ${nglycIndices.length}`,
      detail: 'N-X-[S/T] (X!=P) sequons can be N-glycosylated in mammalian/eukaryotic expression.',
      indices: nglycIndices,
    });
  }

  if (deamIndices.length > 0) {
    liabilities.push({
      kind: 'deamidation',
      severity: 'info',
      label: `NG deamid: ${deamIndices.length}`,
      detail: 'Asn-Gly (NG) motifs are prone to non-enzymatic deamidation over time.',
      indices: deamIndices,
    });
  }

  if (isomIndices.length > 0) {
    liabilities.push({
      kind: 'isomerization',
      severity: 'info',
      label: `DG isom: ${isomIndices.length}`,
      detail: 'Asp-Gly (DG) motifs are prone to aspartate isomerization.',
      indices: isomIndices,
    });
  }

  // Hydrophobic patches (>= 6 consecutive hydrophobic residues)
  const hydroIndices: number[] = [];
  let runStart = -1;
  for (let i = 0; i <= n; i++) {
    if (i < n && HYDROPHOBIC_SET.has(clean[i])) {
      if (runStart === -1) runStart = i;
    } else {
      if (runStart !== -1 && i - runStart >= 6) {
        for (let k = runStart; k < i; k++) hydroIndices.push(k);
      }
      runStart = -1;
    }
  }
  if (hydroIndices.length > 0) {
    liabilities.push({
      kind: 'hydrophobic-patch',
      severity: 'warn',
      label: `Hydrophobic patch (${hydroIndices.length} aa)`,
      detail: 'Run of >=6 consecutive hydrophobic residues may reduce solubility or promote aggregation.',
      indices: hydroIndices,
    });
  }

  return {
    length: n,
    mwKda: Math.round((mwDa / 1000) * 100) / 100,
    pI: Math.round(pI * 100) / 100,
    netCharge74: Math.round(netCharge74 * 10) / 10,
    extCoeff280,
    cysCount: cysIndices.length,
    liabilities,
  };
}

/** Format a clean multi-chain FASTA string (60-char wrapped lines). */
export function formatMultiChainFasta(
  baseId: string,
  chains: ReadonlyArray<{ chainId: string; roleTag?: string; sequence: string }>,
  opts: { colonMultimer?: boolean } = {},
): string {
  const wrap = (seq: string) => seq.replace(/(.{1,60})/g, '$1\n').trimEnd();
  if (opts.colonMultimer) {
    const joined = chains.map((c) => c.sequence.trim()).filter(Boolean).join(':');
    return `>${baseId}\n${wrap(joined)}`;
  }
  return chains
    .filter((c) => c.sequence.trim().length > 0)
    .map((c) => {
      const role = c.roleTag ? `|${c.roleTag}` : '';
      return `>${baseId}|chain_${c.chainId}${role}\n${wrap(c.sequence.trim())}`;
    })
    .join('\n');
}

/** Extract clean 20-AA order-ready sequence (strips separators/X/non-canonical chars). */
export function formatOrderReadyAa(sequence: string, ensureStartMet = false): string {
  const clean = sequence.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
  if (!clean) return '';
  if (ensureStartMet && !clean.startsWith('M')) return `M${clean}`;
  return clean;
}
