// Layer 3 Plugin CPU Runner — top-k-filter
// Deterministic per-backbone or global top-K sequence ranker & filter.

import { sha256Hex, type CollectionMember } from '$lib/engine/collections';
import { planTopKFilter } from '$lib/engine/work-plan';
import { canonicalJsonStringify, fnv1a } from '$lib/kernel/hash-utils';
import type {
  CanonicalArtifact,
  CollectionItemOutcome,
  ExcludedCandidateRecord,
  SequenceCandidate,
  SequenceSet,
  WorkflowNode,
} from '$lib/engine/types';

export async function runTopKFilterCpu(
  node: WorkflowNode,
  inputs: Record<string, CanonicalArtifact>,
): Promise<Record<string, CanonicalArtifact>> {
  const seqIn = inputs.sequences as SequenceSet | undefined;
  if (!seqIn || seqIn.kind !== 'SequenceSet') {
    throw new Error('top-k-filter requires an upstream SequenceSet input');
  }
  const modeRaw = String(node.params.mode ?? 'per-backbone');
  const mode: 'per-backbone' | 'global' = modeRaw === 'global' ? 'global' : 'per-backbone';
  const rawTopK = Number(node.params.topK ?? 2);
  const topK = Number.isFinite(rawTopK) ? Math.max(1, Math.min(16, Math.round(rawTopK))) : 2;
  const rawMaxTotal = Number(node.params.maxTotal ?? 8);
  const maxTotal = Number.isFinite(rawMaxTotal)
    ? Math.max(1, Math.min(64, Math.round(rawMaxTotal)))
    : 8;
  const metricRaw = String(node.params.metric ?? 'score');
  const metric: 'score' | 'globalScore' = metricRaw === 'globalScore' ? 'globalScore' : 'score';
  const directionRaw = String(node.params.direction ?? 'min');
  const direction: 'asc' | 'desc' =
    directionRaw === 'max' || directionRaw === 'desc' ? 'desc' : 'asc';
  const directionParam: 'min' | 'max' = direction === 'desc' ? 'max' : 'min';

  const setParent = (seqIn.sourceStructureId ?? seqIn.parentStructureId ?? '').trim();
  const hasSetParent = setParent.length > 0 && setParent.toLowerCase() !== 'none';
  const groupOrder = new Map<string, number>();
  const origIndexByKey = new Map<string, number>();
  const origCandByKey = new Map<string, SequenceCandidate>();
  const usedKeys = new Set<string>();

  const members: CollectionMember<SequenceCandidate>[] = seqIn.candidates.map((cand, idx) => {
    const rawDesignAnchor = (
      cand.designKey ??
      cand.sourceStructureId ??
      (hasSetParent ? setParent : '')
    ).trim();
    const hasDesign = rawDesignAnchor.length > 0 && rawDesignAnchor.toLowerCase() !== 'none';
    const designKey = hasDesign ? rawDesignAnchor : undefined;
    let designIndex = cand.designIndex;
    if (designKey !== undefined && designIndex === undefined) {
      if (!groupOrder.has(designKey)) groupOrder.set(designKey, groupOrder.size);
      designIndex = groupOrder.get(designKey);
    }
    let memberKey = (cand.sequenceKey ?? cand.key ?? cand.id).trim() || `seq-${idx}`;
    if (usedKeys.has(memberKey)) {
      memberKey = `${memberKey}::${idx}`;
    }
    usedKeys.add(memberKey);
    origIndexByKey.set(memberKey, idx);
    origCandByKey.set(memberKey, cand);
    const ordinal =
      typeof cand.sampleIndex === 'number' && Number.isInteger(cand.sampleIndex) && cand.sampleIndex >= 0
        ? cand.sampleIndex
        : idx;
    const candSourceId = (cand.sourceStructureId ?? (hasSetParent ? setParent : '')).trim();
    const hasSourceId = candSourceId.length > 0 && candSourceId.toLowerCase() !== 'none';
    return {
      key: memberKey,
      ordinal,
      memberKind: 'SequenceCandidate',
      lineage: {
        ...(designKey ? { designKey, designIndex } : {}),
        sequenceKey: memberKey,
        sequenceIndex: ordinal,
        ...(hasSourceId
          ? {
              sourceStructure: {
                artifactId: candSourceId,
                payloadDigest: '',
                ...(designKey ? { designKey } : {}),
              },
            }
          : {}),
        parentKeys: designKey ? [designKey] : [],
      },
      payload: cand,
      summary: {
        label: cand.id,
        residueCount: cand.sequence.length,
        sequenceDigest: sha256Hex(cand.sequence),
        score: cand.score,
        globalScore: cand.globalScore,
      },
    };
  });

  const plan = planTopKFilter(members, {
    mode,
    topK,
    maxTotal,
    metric,
    direction,
  });

  const selectedCandidates: SequenceCandidate[] = plan.selected.map((s) => s.member.payload!);
  const excludedCandidates: ExcludedCandidateRecord[] = plan.excluded.map((ex, exIdx) => {
    const orig = origCandByKey.get(ex.key);
    return {
      key: ex.key,
      id: orig?.id ?? ex.key,
      index: origIndexByKey.get(ex.key) ?? selectedCandidates.length + exIdx,
      reason: ex.reason ?? `Excluded by top-k-filter (${mode}, topK=${topK}, maxTotal=${maxTotal})`,
    };
  });
  const itemOutcomes: CollectionItemOutcome[] = [
    ...plan.selected.map((s): CollectionItemOutcome => ({
      key: s.key,
      status: 'completed',
    })),
    ...excludedCandidates.map((ex): CollectionItemOutcome => ({
      key: ex.key,
      status: 'excluded',
      reason: ex.reason,
    })),
  ];

  const distinctSelectedSources = [
    ...new Set(
      selectedCandidates
        .map((c) => c.sourceStructureId?.trim() ?? '')
        .filter((id) => id.length > 0 && id.toLowerCase() !== 'none'),
    ),
  ];

  const filteredSet: SequenceSet = {
    kind: 'SequenceSet',
    id: `topk-${seqIn.id}-${fnv1a(canonicalJsonStringify({ mode, topK, maxTotal, metric, direction, planDigest: plan.planDigest }))}`,
    parentStructureId: seqIn.parentStructureId,
    ...(seqIn.sourceStructureId ? { sourceStructureId: seqIn.sourceStructureId } : {}),
    ...(distinctSelectedSources.length > 1
      ? { sourceStructureIds: distinctSelectedSources }
      : seqIn.sourceStructureIds
        ? { sourceStructureIds: [...seqIn.sourceStructureIds] }
        : {}),
    sourceModelId: seqIn.sourceModelId,
    ...(seqIn.collectionKind ? { collectionKind: seqIn.collectionKind } : {}),
    ...(seqIn.lockedChains ? { lockedChains: [...seqIn.lockedChains] } : {}),
    ...(seqIn.chainRoles ? { chainRoles: { ...seqIn.chainRoles } } : {}),
    ...(excludedCandidates.length > 0 ? { excludedCandidates } : {}),
    itemOutcomes,
    filterMetadata: {
      mode,
      topK,
      maxTotal: plan.policy.maxTotal,
      metric,
      direction: directionParam,
      totalInput: plan.summary.totalInput,
      selectedCount: plan.summary.selectedCount,
      excludedCount: plan.summary.excludedCount,
      description: plan.summary.description,
    },
    candidates: selectedCandidates,
  };
  return { sequences: filteredSet };
}
