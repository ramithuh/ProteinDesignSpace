import type { SequenceCandidate } from '$lib/engine/types';
import { ESMFOLD2_STEPS } from './registry';

export function esmfold2ParamIssue(params: Record<string, unknown>): string | undefined {
  const steps = params.esmfold2Steps;
  if (steps !== undefined && !ESMFOLD2_STEPS.some((n) => steps === n || steps === String(n))) {
    return `ESMFold2 diffusion schedule must be one of ${ESMFOLD2_STEPS.join(', ')}.`;
  }
  if ((params.msaMode ?? 'single') !== 'single') {
    return 'ESMFold2 requires single-sequence mode.';
  }
}

/** Preserve both explicit chain lengths and inline chain boundaries at the worker boundary. */
export function esmfold2Chains(candidate: SequenceCandidate): string[] {
  const parts = candidate.sequence.replace(/\s/g, '').toUpperCase().split(/[:/]/);
  if (parts.some((part) => !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(part))) {
    throw new Error(`ESMFold2 needs canonical protein residues in "${candidate.id}".`);
  }
  const sequence = parts.join('');
  if (sequence.length < 5 || sequence.length > 512) {
    throw new Error(`ESMFold2 supports 5–512 protein residues per candidate ("${candidate.id}" has ${sequence.length}).`);
  }
  const lengths = candidate.chainLengths ?? parts.map((part) => part.length);
  if (!Array.isArray(lengths) || !lengths.length || lengths.length > 62 ||
      lengths.some((n) => !Number.isInteger(n) || n <= 0) ||
      lengths.reduce((a, b) => a + b, 0) !== sequence.length ||
      (parts.length > 1 && (parts.length !== lengths.length || parts.some((p, i) => p.length !== lengths[i])))) {
    throw new Error(`ESMFold2 received invalid chain boundaries for "${candidate.id}".`);
  }
  let offset = 0;
  return lengths.map((n) => {
    const chain = sequence.slice(offset, offset + n);
    offset += n;
    return chain;
  });
}
