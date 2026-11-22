export interface TrackTooltipPositionInput {
  svgX: number;
  viewBoxWidth: number;
  containerWidthPx?: number | null;
  tooltipWidthPx?: number | null;
  textLength?: number;
  edgePaddingPx?: number;
}

export interface TrackTooltipPosition {
  align: 'left' | 'center' | 'right';
  left: string;
  right: string;
  transform: string;
  pctX: number;
}

/**
 * Compute non-clipping horizontal tooltip placement inside a relative `overflow: hidden`
 * chart container. Snaps to `left: <pad>px` near the left edge and `right: <pad>px` near
 * the right edge; otherwise centers with `translateX(-50%)`.
 */
export function computeTrackTooltipPosition(
  input: TrackTooltipPositionInput,
): TrackTooltipPosition {
  const edgePaddingPx = input.edgePaddingPx ?? 6;
  const vbW = Math.max(1, input.viewBoxWidth);
  const pctX = Math.max(0, Math.min(100, (input.svgX / vbW) * 100));
  const estW = Math.max(input.tooltipWidthPx ?? 0, (input.textLength ?? 18) * 6.8 + 18);
  const boxW =
    input.containerWidthPx && input.containerWidthPx > 0 ? input.containerWidthPx : 360;
  // Match .rt-tooltip `max-width: calc(100% - 12px)` so wide labels snap cleanly to left/right
  // edges instead of entering the center zone and clipping past [0, containerWidthPx].
  const effectiveW = Math.min(estW, Math.max(0, boxW - 2 * edgePaddingPx));
  const halfPct = Math.min(50, Math.max(12, ((effectiveW * 0.5 + edgePaddingPx) / boxW) * 100));

  if (pctX <= halfPct) {
    return {
      align: 'left',
      left: `${edgePaddingPx}px`,
      right: 'auto',
      transform: 'translateX(0)',
      pctX,
    };
  }
  if (pctX >= 100 - halfPct) {
    return {
      align: 'right',
      left: 'auto',
      right: `${edgePaddingPx}px`,
      transform: 'translateX(0)',
      pctX,
    };
  }
  return {
    align: 'center',
    left: `${pctX.toFixed(2)}%`,
    right: 'auto',
    transform: 'translateX(-50%)',
    pctX,
  };
}
