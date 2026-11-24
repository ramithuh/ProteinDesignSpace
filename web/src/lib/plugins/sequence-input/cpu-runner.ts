// Layer 3 Plugin CPU Runner — sequence-input
// Pure CPU materializer shared by ExecutionScheduler and instant live preview.

import { validateSequenceText } from '$lib/engine/param-validation';
import { splitMultiChainSequence } from '$lib/engine/bio-capabilities';
import { fnv1a } from '$lib/kernel/hash-utils';
import type { CanonicalArtifact, SequenceSet, WorkflowNode } from '$lib/engine/types';

/**
 * Pure CPU materializer for a `sequence-input` node.
 * Shared by both ExecutionScheduler and instant live preview (no Run needed).
 */
export function materializeSequenceInputNode(node: WorkflowNode): SequenceSet {
  const raw = String(node.params.sequence ?? '').trim();
  if (!raw) throw new Error('Sequence Input is empty');
  const seqCheck = validateSequenceText(raw);
  if (seqCheck.status === 'error') {
    throw new Error(seqCheck.message);
  }
  const candidates: Array<{
    id: string;
    key: string;
    sequence: string;
    chainLengths?: number[];
    score: null;
    globalScore: null;
    sourceModelId: string;
    sourceStructureId: string;
  }> = [];
  if (raw.includes('>')) {
    const blocks = raw.split(/^\s*>/m).filter((b) => b.trim().length > 0);
    const seenIds = new Set<string>();
    blocks.forEach((b, idx) => {
      const lines = b.split('\n');
      const baseHeader = lines[0].trim().split(/\s+/)[0] || `seq-${idx + 1}`;
      const split = splitMultiChainSequence(lines.slice(1).join(''));
      if (split.concatenated.length >= 5) {
        let id = baseHeader;
        let suffix = 2;
        while (seenIds.has(id)) id = `${baseHeader}_${suffix++}`;
        seenIds.add(id);
        candidates.push({
          id,
          key: id,
          sequence: split.concatenated,
          ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
          score: null,
          globalScore: null,
          sourceModelId: 'sequence-input',
          sourceStructureId: 'none',
        });
      }
    });
  } else {
    const split = splitMultiChainSequence(raw);
    if (split.concatenated.length >= 5) {
      const id = String(node.params.label || 'custom-seq-1');
      candidates.push({
        id,
        key: id,
        sequence: split.concatenated,
        ...(split.chains.length > 1 ? { chainLengths: split.chainLengths } : {}),
        score: null,
        globalScore: null,
        sourceModelId: 'sequence-input',
        sourceStructureId: 'none',
      });
    }
  }
  if (candidates.length === 0) throw new Error('No valid amino acid sequences (>= 5 aa) found');
  return {
    kind: 'SequenceSet',
    id: `seqin-${fnv1a(JSON.stringify(candidates))}`,
    parentStructureId: 'none',
    sourceStructureId: 'none',
    sourceModelId: 'sequence-input',
    collectionKind: 'independent-candidates',
    candidates,
  };
}

export async function runSequenceInputCpu(
  node: WorkflowNode,
): Promise<Record<string, CanonicalArtifact>> {
  const seqSet = materializeSequenceInputNode(node);
  return { sequences: seqSet };
}
