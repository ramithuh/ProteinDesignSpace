// Layer 1 Core Kernel — Domain-Agnostic Keyed Multi-Branch Collection Join.
// Pairs items across arbitrary branches by canonical candidate identity `key`
// (regardless of branch rank order), explicitly marking missing members on any branch
// and preserving confidence-less samples without fabricating confidence scores.
// Strict Layer 1 rule: ZERO imports from Layer 2 (domains/*) or Layer 3 (plugins/*).

export type ComparisonPairingMode = 'by-candidate' | 'branch-best';

export interface KeyedCollectionItem<T = unknown> {
  key: string;
  sequence?: string;
  rank?: number;
  score?: number | null;
  confidence?: number | null;
  confidenceAvailable?: boolean;
  ipsae?: number | null;
  value: T;
}

export interface KeyedCollectionBranch<T = unknown, TCollectionKind extends string = string> {
  branchId: string;
  label?: string;
  collectionKind?: TCollectionKind;
  items: readonly KeyedCollectionItem<T>[];
}

export interface JoinedBranchEntry<T = unknown> {
  branchId: string;
  present: boolean;
  missingReason?: string;
  rank: number | null;
  score: number | null;
  confidence: number | null;
  confidenceAvailable: boolean;
  ipsae?: number | null;
  item: KeyedCollectionItem<T> | null;
}

export interface JoinedCollectionRow<T = unknown> {
  key: string;
  sequence?: string;
  presentCount: number;
  missingBranchIds: string[];
  byBranch: Record<string, JoinedBranchEntry<T>>;
}

export interface KeyedCollectionJoinResult<T = unknown> {
  keys: string[];
  branchIds: string[];
  rows: JoinedCollectionRow<T>[];
  completeRowCount: number;
  partialRowCount: number;
}

/**
 * General keyed multi-branch collection join helper.
 * Pairs items across arbitrary branches by canonical candidate identity `key`
 * (regardless of branch rank order), explicitly marking missing members on any branch
 * and preserving confidence-less samples without fabricating confidence scores.
 */
export function joinKeyedCollections<T = unknown, TCollectionKind extends string = string>(
  branches: readonly KeyedCollectionBranch<T, TCollectionKind>[],
  opts: { matchBySequenceFallback?: boolean } = {},
): KeyedCollectionJoinResult<T> {
  const branchIds = branches.map((b) => b.branchId);
  const matchBySeq = Boolean(opts.matchBySequenceFallback);

  // Disambiguate duplicate keys within the same branch deterministically so multi-sample
  // members sharing a base key are never silently dropped or collapsed.
  const normalizedBranches: KeyedCollectionBranch<T, TCollectionKind>[] = branches.map((br) => {
    const usedKeys = new Set<string>();
    const baseCounts = new Map<string, number>();
    const items = br.items.map((it) => {
      const base = it.key;
      const count = (baseCounts.get(base) ?? 0) + 1;
      baseCounts.set(base, count);
      if (!usedKeys.has(base)) {
        usedKeys.add(base);
        return it;
      }
      let suffix = Math.max(2, count);
      let disambig = `${base}#${suffix}`;
      while (usedKeys.has(disambig)) {
        suffix++;
        disambig = `${base}#${suffix}`;
      }
      usedKeys.add(disambig);
      return { ...it, key: disambig };
    });
    return { ...br, items };
  });

  // Track which item indices in each branch have been matched so sequence fallback never double-binds.
  const consumedByBranch = new Map<string, Set<number>>();
  const keyIndexByBranch = new Map<string, Map<string, number>>();
  const keyBranchCount = new Map<string, number>();
  for (const br of normalizedBranches) {
    consumedByBranch.set(br.branchId, new Set());
    const idxMap = new Map<string, number>();
    const seenKeysInBr = new Set<string>();
    br.items.forEach((it, idx) => {
      if (!idxMap.has(it.key)) idxMap.set(it.key, idx);
      if (!seenKeysInBr.has(it.key)) {
        seenKeysInBr.add(it.key);
        keyBranchCount.set(it.key, (keyBranchCount.get(it.key) ?? 0) + 1);
      }
    });
    keyIndexByBranch.set(br.branchId, idxMap);
  }

  // First determine whether any branch shares exact keys; sequence fallback is only used
  // for items that have no exact-key peer in another branch.
  const rows: JoinedCollectionRow<T>[] = [];
  const keys: string[] = [];
  const seenKeys = new Set<string>();

  const anchorPasses: Array<(key: string) => boolean> = matchBySeq
    ? [(key) => (keyBranchCount.get(key) ?? 0) >= 2, () => true]
    : [() => true];

  for (const allowAnchorKey of anchorPasses) {
    for (const sourceBranch of normalizedBranches) {
      const srcConsumed = consumedByBranch.get(sourceBranch.branchId)!;
      for (let srcIdx = 0; srcIdx < sourceBranch.items.length; srcIdx++) {
        if (srcConsumed.has(srcIdx)) continue;
        const anchorItem = sourceBranch.items[srcIdx];
        const key = anchorItem.key;
        if (!allowAnchorKey(key)) continue;
        if (seenKeys.has(key)) continue;
        seenKeys.add(key);
        keys.push(key);

        const normAnchorSeq = anchorItem.sequence?.trim().toUpperCase() || '';
        const byBranch: Record<string, JoinedBranchEntry<T>> = {};
        const missingBranchIds: string[] = [];
        let presentCount = 0;

        for (const br of normalizedBranches) {
          const consumed = consumedByBranch.get(br.branchId)!;
          let matchIdx: number | undefined = keyIndexByBranch.get(br.branchId)!.get(key);
          if (matchIdx !== undefined && consumed.has(matchIdx)) {
            matchIdx = undefined;
          }
          if (matchIdx === undefined && matchBySeq && normAnchorSeq.length > 0) {
            // Only fallback-match by exact sequence when unique & unconsumed in this branch
            // AND the candidate key does not already have an exact-key peer in another branch.
            const seqMatches: number[] = [];
            for (let j = 0; j < br.items.length; j++) {
              if (consumed.has(j)) continue;
              if ((keyBranchCount.get(br.items[j].key) ?? 0) >= 2) continue;
              const candSeq = br.items[j].sequence?.trim().toUpperCase() || '';
              if (candSeq === normAnchorSeq) seqMatches.push(j);
            }
            if (seqMatches.length === 1) {
              matchIdx = seqMatches[0];
            }
          }

          if (matchIdx !== undefined) {
            consumed.add(matchIdx);
            const item = br.items[matchIdx];
            const hasConf =
              item.confidenceAvailable !== false &&
              typeof item.confidence === 'number' &&
              Number.isFinite(item.confidence);
            byBranch[br.branchId] = {
              branchId: br.branchId,
              present: true,
              rank: item.rank ?? matchIdx + 1,
              score: typeof item.score === 'number' && Number.isFinite(item.score) ? item.score : null,
              confidence: hasConf ? (item.confidence as number) : null,
              confidenceAvailable: hasConf,
              ...(typeof item.ipsae === 'number' && Number.isFinite(item.ipsae)
                ? { ipsae: item.ipsae }
                : { ipsae: null }),
              item,
            };
            presentCount++;
          } else {
            missingBranchIds.push(br.branchId);
            byBranch[br.branchId] = {
              branchId: br.branchId,
              present: false,
              missingReason: `Candidate "${key}" absent in branch "${br.label ?? br.branchId}"`,
              rank: null,
              score: null,
              confidence: null,
              confidenceAvailable: false,
              ipsae: null,
              item: null,
            };
          }
        }

        rows.push({
          key,
          ...(normAnchorSeq ? { sequence: normAnchorSeq } : {}),
          presentCount,
          missingBranchIds,
          byBranch,
        });
      }
    }
  }

  let completeRowCount = 0;
  let partialRowCount = 0;
  for (const r of rows) {
    if (r.missingBranchIds.length === 0) completeRowCount++;
    else partialRowCount++;
  }

  return {
    keys,
    branchIds,
    rows,
    completeRowCount,
    partialRowCount,
  };
}
