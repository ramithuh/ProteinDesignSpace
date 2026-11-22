import type { BfactorSemantics } from '$lib/engine/types';

export type UncertaintyColorArg = {
  color: 'uncertainty';
  colorParams?: { domain: [number, number] };
};

/**
 * Resolve a Mol* `'uncertainty'` (B-factor) color argument from observed polymer Cα
 * (or fallback polymer/all-atom) B-factor values.
 *
 * Ensures full contrast for:
 * - `dynamics-prob` (`0..1` probabilities, ignoring minority crystal B-factor outliers)
 * - `ddg-stability` (raw signed or non-negative kcal/mol values, including `mean-ddg` / `worst-destabilizing`
 *   with `min >= 0` and `max <= 100.5` instead of collapsing into `[0, 100]`)
 * - `llr-fitness` (signed or non-negative ESM-C LLR fitness values around `0`, ignoring crystal B-factor outliers)
 * - `crystal-b` / `undefined` (observed `[min, max]` domain when non-constant)
 */
export function resolveUncertaintyColorArgFromValues(
  values: ArrayLike<number>,
  semantics?: BfactorSemantics | 'llr-fitness',
): UncertaintyColorArg {
  const finite: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (typeof v === 'number' && Number.isFinite(v)) {
      finite.push(v);
    }
  }
  if (finite.length === 0) {
    return { color: 'uncertainty' };
  }

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < finite.length; i++) {
    const v = finite[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  if (semantics === 'dynamics-prob') {
    let unitCount = 0;
    for (let i = 0; i < finite.length; i++) {
      const v = finite[i];
      if (v >= -0.05 && v <= 1.05) unitCount++;
    }
    if (unitCount > 0 && unitCount / finite.length >= 0.5) {
      return { color: 'uncertainty', colorParams: { domain: [0, 1] } };
    }
    return { color: 'uncertainty', colorParams: { domain: [0, 100] } };
  }

  if (semantics === 'ddg-stability') {
    if (min < 0) {
      let effAbsMax = 0;
      let hasSubset = false;
      for (let i = 0; i < finite.length; i++) {
        if (finite[i] <= 15) {
          hasSubset = true;
          const a = Math.abs(finite[i]);
          if (a > effAbsMax) effAbsMax = a;
        }
      }
      if (!hasSubset) {
        effAbsMax = Math.max(Math.abs(min), Math.abs(max));
      }
      const lim = Math.max(1.5, Math.min(2, effAbsMax));
      return { color: 'uncertainty', colorParams: { domain: [-lim, lim] } };
    }

    // Non-negative ΔΔG (e.g. mean-ddg, worst-destabilizing, positive-only best-stabilizing, or constant 0)
    let subCount = 0;
    let subMin = Infinity;
    let subMax = -Infinity;
    for (let i = 0; i < finite.length; i++) {
      const v = finite[i];
      if (v <= 15) {
        subCount++;
        if (v < subMin) subMin = v;
        if (v > subMax) subMax = v;
      }
    }
    const useSub = subCount > 0 && subCount / finite.length >= 0.5;
    const effMin = useSub ? subMin : min;
    const effMax = useSub ? subMax : max;

    if (effMin < 4) {
      const hi = Math.max(1.5, Math.min(4, effMax));
      return { color: 'uncertainty', colorParams: { domain: [0, hi] } };
    }
    if (effMax > effMin) {
      return { color: 'uncertainty', colorParams: { domain: [effMin, effMax] } };
    }
    return { color: 'uncertainty', colorParams: { domain: [0, effMax] } };
  }

  if ((semantics as string) === 'llr-fitness') {
    if (min < 0) {
      let effAbsMax = 0;
      let hasSubset = false;
      for (let i = 0; i < finite.length; i++) {
        if (finite[i] <= 20) {
          hasSubset = true;
          const a = Math.abs(finite[i]);
          if (a > effAbsMax) effAbsMax = a;
        }
      }
      if (!hasSubset) {
        effAbsMax = Math.max(Math.abs(min), Math.abs(max));
      }
      const lim = Math.max(1.5, Math.min(6, effAbsMax));
      return { color: 'uncertainty', colorParams: { domain: [-lim, lim] } };
    }

    let subCount = 0;
    let subMin = Infinity;
    let subMax = -Infinity;
    for (let i = 0; i < finite.length; i++) {
      const v = finite[i];
      if (v <= 20) {
        subCount++;
        if (v < subMin) subMin = v;
        if (v > subMax) subMax = v;
      }
    }
    const useSub = subCount > 0 && subCount / finite.length >= 0.5;
    const effMin = useSub ? subMin : min;
    const effMax = useSub ? subMax : max;

    if (effMin < 6) {
      const hi = Math.max(1.5, Math.min(6, effMax));
      return { color: 'uncertainty', colorParams: { domain: [0, hi] } };
    }
    if (effMax > effMin) {
      return { color: 'uncertainty', colorParams: { domain: [effMin, effMax] } };
    }
    return { color: 'uncertainty', colorParams: { domain: [0, effMax] } };
  }

  // 'crystal-b' or undefined: fit observed domain when non-constant
  if (max > min) {
    return { color: 'uncertainty', colorParams: { domain: [min, max] } };
  }
  return { color: 'uncertainty' };
}
