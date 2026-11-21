import type { PreviewResidue } from './studio-types';

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
