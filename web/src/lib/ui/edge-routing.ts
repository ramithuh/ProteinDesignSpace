// Obstacle-avoiding orthogonal edge router for the denovo.run workflow canvas.
// Uses standard smoothstep when the direct path is clear, and smoothly wraps around
// intermediate node cards (above, below, or through clear corridors) when blocked.

export interface ObstacleRect {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface SmartRouteOptions {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourceId: string;
  targetId: string;
  nodes: readonly ObstacleRect[];
  stub?: number;
  padding?: number;
  borderRadius?: number;
  laneStagger?: number;
}

export interface RoutedEdgeResult {
  path: string;
  labelX: number;
  labelY: number;
  points: Point[];
}

export interface HitRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Compute a non-degenerate 2D hit rectangle for straight axis-aligned edges.
 * Safari/WebKit can fail to hit-test strictly horizontal or vertical SVG `<path>`
 * elements whose geometry bounding box collapses to zero height or width.
 */
export function computeStraightEdgeHitRect(opts: {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  points?: readonly Point[] | null;
  interactionWidth?: number;
}): HitRect | null {
  const { sourceX, sourceY, targetX, targetY, points, interactionWidth = 22 } = opts;
  if (points && points.length > 2) return null;

  const hitW = Math.max(12, interactionWidth);
  const dx = Math.abs(targetX - sourceX);
  const dy = Math.abs(targetY - sourceY);

  if (dy < 4 && dx >= 8) {
    const minX = Math.min(sourceX, targetX);
    const maxX = Math.max(sourceX, targetX);
    const inset = Math.min(4, dx * 0.15);
    const width = Math.max(4, maxX - minX - inset * 2);
    const midY = (sourceY + targetY) * 0.5;
    return {
      x: minX + inset,
      y: midY - hitW * 0.5,
      width,
      height: hitW,
    };
  }

  if (dx < 4 && dy >= 8) {
    const minY = Math.min(sourceY, targetY);
    const maxY = Math.max(sourceY, targetY);
    const inset = Math.min(4, dy * 0.15);
    const height = Math.max(4, maxY - minY - inset * 2);
    const midX = (sourceX + targetX) * 0.5;
    return {
      x: midX - hitW * 0.5,
      y: minY + inset,
      width: hitW,
      height,
    };
  }

  return null;
}

interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function expandRect(r: ObstacleRect, padX: number, padY = padX): Box {
  return {
    left: r.x - padX,
    right: r.x + r.width + padX,
    top: r.y - padY,
    bottom: r.y + r.height + padY,
  };
}

/** True if axis-aligned segment (a -> b) strictly enters box interior. */
export function segmentHitsBox(a: Point, b: Point, box: Box): boolean {
  const eps = 0.5;
  if (Math.abs(a.y - b.y) < eps) {
    // Horizontal segment
    const y = (a.y + b.y) * 0.5;
    if (y <= box.top || y >= box.bottom) return false;
    const minX = Math.min(a.x, b.x);
    const maxX = Math.max(a.x, b.x);
    return maxX > box.left && minX < box.right;
  }
  if (Math.abs(a.x - b.x) < eps) {
    // Vertical segment
    const x = (a.x + b.x) * 0.5;
    if (x <= box.left || x >= box.right) return false;
    const minY = Math.min(a.y, b.y);
    const maxY = Math.max(a.y, b.y);
    return maxY > box.top && minY < box.bottom;
  }
  // Fallback AABB overlap for non-orthogonal segments
  return (
    Math.max(a.x, b.x) > box.left &&
    Math.min(a.x, b.x) < box.right &&
    Math.max(a.y, b.y) > box.top &&
    Math.min(a.y, b.y) < box.bottom
  );
}

export function countPolylineHits(pts: readonly Point[], boxes: readonly Box[]): number {
  let hits = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (Math.abs(a.x - b.x) < 0.25 && Math.abs(a.y - b.y) < 0.25) continue;
    for (let j = 0; j < boxes.length; j++) {
      if (segmentHitsBox(a, b, boxes[j])) hits++;
    }
  }
  return hits;
}

export function cleanPolyline(pts: readonly Point[]): Point[] {
  const dedup: Point[] = [];
  for (const p of pts) {
    const prev = dedup[dedup.length - 1];
    if (!prev || Math.hypot(p.x - prev.x, p.y - prev.y) >= 0.5) {
      dedup.push({ x: p.x, y: p.y });
    }
  }
  if (dedup.length <= 2) return dedup;

  const out: Point[] = [dedup[0]];
  for (let i = 1; i < dedup.length - 1; i++) {
    const prev = out[out.length - 1];
    const cur = dedup[i];
    const next = dedup[i + 1];
    const sameX = Math.abs(prev.x - cur.x) < 0.5 && Math.abs(cur.x - next.x) < 0.5;
    const sameY = Math.abs(prev.y - cur.y) < 0.5 && Math.abs(cur.y - next.y) < 0.5;
    if (!sameX && !sameY) {
      out.push(cur);
    }
  }
  out.push(dedup[dedup.length - 1]);
  return out;
}

export function polylineLength(pts: readonly Point[]): number {
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    total += Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
  }
  return total;
}

/** Convert an orthogonal polyline into a smooth SVG path with quadratic-bezier corners. */
export function pointsToRoundedPath(rawPts: readonly Point[], radius = 12): RoutedEdgeResult {
  const pts = cleanPolyline(rawPts);
  if (pts.length === 0) {
    return { path: '', labelX: 0, labelY: 0, points: [] };
  }
  if (pts.length === 1) {
    return {
      path: `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`,
      labelX: pts[0].x,
      labelY: pts[0].y,
      points: pts,
    };
  }

  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const prev = pts[i - 1];
    const cur = pts[i];
    const next = pts[i + 1];
    const dx1 = prev.x - cur.x;
    const dy1 = prev.y - cur.y;
    const len1 = Math.hypot(dx1, dy1);
    const dx2 = next.x - cur.x;
    const dy2 = next.y - cur.y;
    const len2 = Math.hypot(dx2, dy2);
    const r = Math.min(radius, len1 * 0.5, len2 * 0.5);
    if (r < 0.5 || len1 < 0.5 || len2 < 0.5) {
      d += ` L ${cur.x.toFixed(1)} ${cur.y.toFixed(1)}`;
    } else {
      const c1x = cur.x + (dx1 / len1) * r;
      const c1y = cur.y + (dy1 / len1) * r;
      const c2x = cur.x + (dx2 / len2) * r;
      const c2y = cur.y + (dy2 / len2) * r;
      d += ` L ${c1x.toFixed(1)} ${c1y.toFixed(1)} Q ${cur.x.toFixed(1)} ${cur.y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)}`;
    }
  }
  const last = pts[pts.length - 1];
  d += ` L ${last.x.toFixed(1)} ${last.y.toFixed(1)}`;

  // Label position at midpoint of the longest segment
  let bestLen = -1;
  let labelX = (pts[0].x + last.x) * 0.5;
  let labelY = (pts[0].y + last.y) * 0.5;
  for (let i = 0; i < pts.length - 1; i++) {
    const segLen = Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
    if (segLen > bestLen) {
      bestLen = segLen;
      labelX = (pts[i].x + pts[i + 1].x) * 0.5;
      labelY = (pts[i].y + pts[i + 1].y) * 0.5;
    }
  }

  return { path: d, labelX, labelY, points: pts };
}

function mergeIntervals(intervals: Array<[number, number]>): Array<[number, number]> {
  if (intervals.length <= 1) return intervals;
  const sorted = [...intervals].sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const cur = sorted[i];
    const last = merged[merged.length - 1];
    if (cur[0] <= last[1]) {
      last[1] = Math.max(last[1], cur[1]);
    } else {
      merged.push([cur[0], cur[1]]);
    }
  }
  return merged;
}

function uniqueSorted(vals: number[], tol = 1): number[] {
  const s = [...vals].sort((a, b) => a - b);
  const out: number[] = [];
  for (const v of s) {
    if (out.length === 0 || Math.abs(v - out[out.length - 1]) > tol) {
      out.push(v);
    }
  }
  return out;
}

/**
 * Compute an obstacle-avoiding orthogonal SVG path for an edge.
 * Returns `null` when the default smoothstep path is unobstructed (caller can use native getSmoothStepPath).
 */
export function computeSmartEdgePath(opts: SmartRouteOptions): RoutedEdgeResult | null {
  const {
    sourceX: sx,
    sourceY: sy,
    targetX: tx,
    targetY: ty,
    sourceId,
    targetId,
    nodes,
    stub = 24,
    padding = 18,
    borderRadius = 12,
    laneStagger = 0,
  } = opts;

  const otherNodes = nodes.filter((n) => n.id !== sourceId && n.id !== targetId);
  const srcNode = nodes.find((n) => n.id === sourceId);
  const dstNode = nodes.find((n) => n.id === targetId);

  const paddedBoxes = otherNodes.map((n) => expandRect(n, padding));
  // Softer horizontal padding for threading tight card-to-card vertical gaps
  const tightBoxes = otherNodes.map((n) => expandRect(n, 6, padding));

  const forward = tx >= sx + 16;

  // 1. Fast path: check standard 3-segment smoothstep Z-path
  if (forward) {
    const midX = (sx + tx) * 0.5;
    const defaultPts: Point[] =
      Math.abs(sy - ty) < 1
        ? [
            { x: sx, y: sy },
            { x: tx, y: ty },
          ]
        : [
            { x: sx, y: sy },
            { x: midX, y: sy },
            { x: midX, y: ty },
            { x: tx, y: ty },
          ];

    if (paddedBoxes.length === 0 || countPolylineHits(defaultPts, paddedBoxes) === 0) {
      return null;
    }

    // 2. Try shifted 3-segment Z-paths (only 2 bends) through clear horizontal gaps
    const minMx = sx + Math.min(stub, (tx - sx) * 0.35);
    const maxMx = tx - Math.min(stub, (tx - sx) * 0.35);
    if (maxMx >= minMx) {
      const xIntervals = mergeIntervals(
        paddedBoxes
          .filter((b) => b.right > minMx && b.left < maxMx)
          .map((b) => [Math.max(minMx, b.left), Math.min(maxMx, b.right)] as [number, number]),
      );

      const zCandidates: number[] = [sx + stub, tx - stub];
      let cursor = minMx;
      for (const [l, r] of xIntervals) {
        if (l - cursor >= 4) {
          zCandidates.push((cursor + l) * 0.5);
        }
        cursor = Math.max(cursor, r);
      }
      if (maxMx - cursor >= 4) {
        zCandidates.push((cursor + maxMx) * 0.5);
      }

      let bestZ: { pts: Point[]; score: number } | null = null;
      for (const mx of uniqueSorted(zCandidates)) {
        if (mx < minMx - 0.5 || mx > maxMx + 0.5) continue;
        const pts = cleanPolyline([
          { x: sx, y: sy },
          { x: mx, y: sy },
          { x: mx, y: ty },
          { x: tx, y: ty },
        ]);
        if (countPolylineHits(pts, paddedBoxes) === 0) {
          const score = polylineLength(pts) + Math.abs(mx - midX) * 0.15;
          if (!bestZ || score < bestZ.score) {
            bestZ = { pts, score };
          }
        }
      }
      if (bestZ) {
        return pointsToRoundedPath(bestZ.pts, borderRadius);
      }
    }

    // 3. 5-segment wrap-around detour (above, below, or through a clear horizontal corridor)
    const spanBoxes = paddedBoxes.filter((b) => b.right > sx + 2 && b.left < tx - 2);
    const allRefBoxes = spanBoxes.length > 0 ? spanBoxes : paddedBoxes;

    const exitXs: number[] = [sx + stub];
    const enterXs: number[] = [tx - stub];

    // If an obstacle sits right next to source or target, also allow threading the physical gap
    let nearestRightOfSrc = Infinity;
    let nearestLeftOfDst = -Infinity;
    for (const n of otherNodes) {
      if (n.x >= sx - 2 && n.x < nearestRightOfSrc) nearestRightOfSrc = n.x;
      const nr = n.x + n.width;
      if (nr <= tx + 2 && nr > nearestLeftOfDst) nearestLeftOfDst = nr;
    }
    if (nearestRightOfSrc < sx + stub + padding && nearestRightOfSrc > sx + 8) {
      exitXs.push(Math.max(sx + 8, (sx + nearestRightOfSrc) * 0.5));
    }
    if (nearestLeftOfDst > tx - stub - padding && nearestLeftOfDst < tx - 8) {
      enterXs.push(Math.min(tx - 8, (nearestLeftOfDst + tx) * 0.5));
    }

    const yIntervals = mergeIntervals(allRefBoxes.map((b) => [b.top, b.bottom] as [number, number]));
    const allYIntervals = mergeIntervals(paddedBoxes.map((b) => [b.top, b.bottom] as [number, number]));

    const laneYs: number[] = [];
    const margin = 6;
    if (yIntervals.length > 0) {
      // Clean lane above and below the blocking cluster
      laneYs.push(yIntervals[0][0] - margin - laneStagger);
      laneYs.push(yIntervals[yIntervals.length - 1][1] + margin + laneStagger);
      // Clear horizontal corridors between stacked rows of nodes
      for (let i = 0; i < yIntervals.length - 1; i++) {
        const gapTop = yIntervals[i][1];
        const gapBot = yIntervals[i + 1][0];
        if (gapBot - gapTop >= 8) {
          const mid = (gapTop + gapBot) * 0.5;
          const clampedShift = Math.min(laneStagger, Math.max(0, (gapBot - gapTop) * 0.25));
          laneYs.push(mid + clampedShift);
        }
      }
    }
    if (allYIntervals.length > 0) {
      // Global outer lanes as guaranteed obstacle-free fallback
      laneYs.push(allYIntervals[0][0] - margin - laneStagger);
      laneYs.push(allYIntervals[allYIntervals.length - 1][1] + margin + laneStagger);
    }

    let best: { pts: Point[]; score: number } | null = null;
    for (const ex of uniqueSorted(exitXs)) {
      for (const nx of uniqueSorted(enterXs)) {
        if (nx <= ex + 2) continue;
        for (const ly of uniqueSorted(laneYs)) {
          const pts = cleanPolyline([
            { x: sx, y: sy },
            { x: ex, y: sy },
            { x: ex, y: ly },
            { x: nx, y: ly },
            { x: nx, y: ty },
            { x: tx, y: ty },
          ]);
          const hardHits = countPolylineHits(pts, tightBoxes);
          const softHits = countPolylineHits(pts, paddedBoxes);
          const score = hardHits * 1_000_000 + softHits * 10_000 + polylineLength(pts);
          if (!best || score < best.score) {
            best = { pts, score };
          }
        }
      }
    }

    if (best) {
      return pointsToRoundedPath(best.pts, borderRadius);
    }
    return null;
  }

  // 4. Backward or vertically tight edge (tx < sx + 16): wrap cleanly around source, target, and obstacles
  const exitX = Math.max(sx + stub, srcNode ? srcNode.x + srcNode.width + padding : sx + stub);
  const enterX = Math.min(tx - stub, dstNode ? dstNode.x - padding : tx - stub);

  const selfBoxes: Box[] = [];
  if (srcNode) selfBoxes.push(expandRect(srcNode, 6, padding));
  if (dstNode) selfBoxes.push(expandRect(dstNode, 6, padding));

  const involvedBoxes = [
    ...paddedBoxes.filter((b) => b.right > enterX - 4 && b.left < exitX + 4),
    ...(srcNode ? [expandRect(srcNode, padding)] : []),
    ...(dstNode ? [expandRect(dstNode, padding)] : []),
  ];

  const yIntervals = mergeIntervals(involvedBoxes.map((b) => [b.top, b.bottom] as [number, number]));
  const laneYs: number[] = [];
  const margin = 6;
  if (yIntervals.length > 0) {
    laneYs.push(yIntervals[0][0] - margin - laneStagger);
    laneYs.push(yIntervals[yIntervals.length - 1][1] + margin + laneStagger);
    for (let i = 0; i < yIntervals.length - 1; i++) {
      const gapTop = yIntervals[i][1];
      const gapBot = yIntervals[i + 1][0];
      if (gapBot - gapTop >= 8) {
        laneYs.push((gapTop + gapBot) * 0.5);
      }
    }
  } else {
    laneYs.push((sy + ty) * 0.5);
  }

  let bestBack: { pts: Point[]; score: number } | null = null;
  for (const ly of uniqueSorted(laneYs)) {
    const pts = cleanPolyline([
      { x: sx, y: sy },
      { x: exitX, y: sy },
      { x: exitX, y: ly },
      { x: enterX, y: ly },
      { x: enterX, y: ty },
      { x: tx, y: ty },
    ]);
    // Check intermediate obstacles on all segments, and source/target cards on the middle traverse segment
    const obstHits = countPolylineHits(pts, tightBoxes);
    const midHits =
      pts.length >= 4 ? countPolylineHits([pts[2], pts[3]], selfBoxes) : 0;
    const score = (obstHits + midHits) * 1_000_000 + polylineLength(pts);
    if (!bestBack || score < bestBack.score) {
      bestBack = { pts, score };
    }
  }

  return bestBack ? pointsToRoundedPath(bestBack.pts, borderRadius) : null;
}
