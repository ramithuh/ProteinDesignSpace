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
 * - `llr-fitness` (adaptive ESM-C LLR fitness range with reversed `[hi, lo]` domain so low/deleterious
 *   paints red and high/beneficial paints blue in Mol*'s `'uncertainty'` theme, matching MutationHeatmap)
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
    // Filter out minority crystal B-factor outliers (> 20) from unpainted chains/HETATMs.
    const inRange: number[] = [];
    for (let i = 0; i < finite.length; i++) {
      if (finite[i] <= 20) inRange.push(finite[i]);
    }
    const sample =
      inRange.length > 0 && inRange.length / finite.length >= 0.5 ? inRange : finite;
    const sorted = [...sample].sort((a, b) => a - b);
    const n = sorted.length;
    const obsLo = n >= 20 ? sorted[Math.floor((n - 1) * 0.05)] : sorted[0];
    const obsHi = n >= 20 ? sorted[Math.ceil((n - 1) * 0.95)] : sorted[n - 1];

    const MIN_SPAN = 2.0;
    let lo: number;
    let hi: number;
    if (obsLo < 0) {
      // Natural proteins have predominantly negative mean LLR (e.g. -14..-1):
      // anchor upper bound at >= 0 (WT-neutral) and preserve full negative conservation
      // range instead of clamping to [-6, +6].
      lo = obsLo;
      hi = Math.max(0, obsHi);
      if (hi - lo < MIN_SPAN) {
        if (obsHi <= 0) {
          lo = hi - MIN_SPAN;
        } else {
          const pad = (MIN_SPAN - (hi - lo)) / 2;
          lo -= pad;
          hi += pad;
        }
      }
    } else {
      // Non-negative LLR values (e.g. synthetic or beneficial-only slices)
      lo = obsLo < 2 ? 0 : obsLo;
      hi = Math.max(lo + MIN_SPAN, obsHi);
    }

    // Reversed domain [hi, lo] for Mol*'s 'uncertainty' theme (red-white-blue with reverse:true):
    // maps low/deleterious LLR -> red (0xBF2222) and high/beneficial LLR -> blue (0x3361E1),
    // matching MutationHeatmap polarity 1:1.
    return { color: 'uncertainty', colorParams: { domain: [hi, lo] } };
  }

  // 'crystal-b' or undefined: fit observed domain when non-constant
  if (max > min) {
    return { color: 'uncertainty', colorParams: { domain: [min, max] } };
  }
  return { color: 'uncertainty' };
}
