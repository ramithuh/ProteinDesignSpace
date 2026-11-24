import { fnv1a } from '$lib/kernel';
import type {
  CollectionKind,
  RepresentativeSelection,
  StructureArtifact,
} from '$lib/engine/types';

export type GninaTaskMode = 'score' | 'pocket-scan' | 'minimize' | 'dock';
export type GninaSortMetric = 'cnnScore' | 'cnnVS' | 'cnnAffinity' | 'vina';

/**
 * Unpack a StructureArtifact into its individual member items when it carries a
 * multi-structure `collection`, or `[struct]` for a single structure.
 */
export function extractStructureCollectionItems(struct: StructureArtifact): StructureArtifact[] {
  if (struct.collection?.items && struct.collection.items.length > 1) {
    return struct.collection.items;
  }
  return [struct];
}

function readMetricScore(
  item: StructureArtifact,
  metric: GninaSortMetric,
): number | null {
  const row = item.dockingMetrics?.rows?.[0];
  if (!row) return null;
  const val =
    metric === 'vina'
      ? row.vinaScore
      : metric === 'cnnAffinity'
        ? row.cnnAffinity
        : metric === 'cnnScore'
          ? row.cnnScore
          : row.cnnVS;
  return typeof val === 'number' && Number.isFinite(val) ? val : null;
}

export function selectBestGninaMember(
  items: readonly StructureArtifact[],
  taskMode: GninaTaskMode,
  sortBy: GninaSortMetric,
): { index: number; selection: RepresentativeSelection } {
  const metric: GninaSortMetric = taskMode === 'dock' ? sortBy : 'cnnVS';
  const lowerIsBetter = metric === 'vina';
  const criterion =
    metric === 'vina'
      ? 'lowest-vina-score'
      : metric === 'cnnAffinity'
        ? 'highest-cnn-affinity'
        : metric === 'cnnScore'
          ? 'highest-cnn-score'
          : 'highest-cnn-vs';

  let bestIdx = 0;
  let bestVal: number | null = null;
  let bestTie: number | null = null;

  for (let i = 0; i < items.length; i++) {
    const val = readMetricScore(items[i], metric);
    const tie = readMetricScore(items[i], 'cnnScore');
    if (val === null) continue;
    if (bestVal === null) {
      bestIdx = i;
      bestVal = val;
      bestTie = tie;
      continue;
    }
    const better = lowerIsBetter ? val < bestVal : val > bestVal;
    const tiedAndBetterTie =
      val === bestVal && tie !== null && (bestTie === null || tie > bestTie);
    if (better || tiedAndBetterTie) {
      bestIdx = i;
      bestVal = val;
      bestTie = tie;
    }
  }

  const chosen = items[bestIdx];
  return {
    index: bestIdx,
    selection: {
      selectedKey: chosen?.designKey ?? chosen?.id ?? `item-${bestIdx}`,
      criterion,
      totalMembers: items.length,
      selectedIndex: bestIdx,
      scoreValue: bestVal,
    },
  };
}

/**
 * Package 1..N GNINA-scored StructureArtifacts into either a single StructureArtifact
 * (N === 1) or a flat `denovo.collection/v1` envelope with the top-scoring fold as
 * representative (N > 1). Strips any nested `.collection` from member items.
 */
export function wrapGninaCollection(
  scoredItems: StructureArtifact[],
  opts: {
    taskMode: GninaTaskMode;
    sortBy: GninaSortMetric;
    collectionKind?: CollectionKind;
  },
): { structure: StructureArtifact; representativeIndex: number } {
  if (scoredItems.length === 0) {
    throw new Error('wrapGninaCollection requires at least one scored structure.');
  }
  const cleanItems: StructureArtifact[] = scoredItems.map((item) => {
    if (!item.collection) return item;
    const copy = { ...item };
    delete copy.collection;
    return copy;
  });

  if (cleanItems.length === 1) {
    return { structure: cleanItems[0], representativeIndex: 0 };
  }

  const { index: bestIdx, selection } = selectBestGninaMember(
    cleanItems,
    opts.taskMode,
    opts.sortBy,
  );
  const primary = cleanItems[bestIdx];
  const batchHash = fnv1a(`${cleanItems.map((it) => it.id).join('|')}:${cleanItems.length}`);
  const keys = cleanItems.map((it, idx) => it.designKey ?? it.id ?? `item-${idx}`);

  const structure: StructureArtifact = {
    ...primary,
    id: `gnina-batch-${opts.taskMode}-${cleanItems.length}-${batchHash}`,
    representativeSelection: selection,
    collection: {
      kind: 'Collection',
      schema: 'denovo.collection/v1',
      id: `gnina-col-${opts.taskMode}-${batchHash}`,
      familyKey: `gnina-${opts.taskMode}-${batchHash}`,
      collectionKind: opts.collectionKind ?? 'independent-candidates',
      requestedCount: cleanItems.length,
      complete: true,
      items: cleanItems,
      keys,
      representativeSelection: selection,
    },
  };

  return { structure, representativeIndex: bestIdx };
}
