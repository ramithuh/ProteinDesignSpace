import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  cleanPolyline,
  computeSmartEdgePath,
  computeStraightEdgeHitRect,
  countPolylineHits,
  pointsToRoundedPath,
  type ObstacleRect,
} from '../ui/edge-routing';

describe('obstacle-avoiding smart edge routing', () => {
  it('returns null (fast-path to smoothstep) when no obstacle blocks the direct wire', () => {
    const nodes: ObstacleRect[] = [
      { id: 'a', x: 40, y: 100, width: 220, height: 150 },
      { id: 'b', x: 340, y: 100, width: 220, height: 150 },
    ];
    const res = computeSmartEdgePath({
      sourceX: 260,
      sourceY: 170,
      targetX: 340,
      targetY: 170,
      sourceId: 'a',
      targetId: 'b',
      nodes,
    });
    expect(res).toBeNull();
  });

  it('wraps cleanly underneath blocking intermediate cards (RFD4 -> scRMSD past MPNN & LocalFold)', () => {
    // Mirrors the exact user screenshot layout:
    // RFD4 slightly lower on the left; ProteinMPNN and LocalFold in the middle; scRMSD on the right
    const nodes: ObstacleRect[] = [
      { id: 'rfd4', x: 40, y: 150, width: 230, height: 170 },
      { id: 'mpnn', x: 360, y: 95, width: 230, height: 140 },
      { id: 'fold', x: 650, y: 85, width: 230, height: 140 },
      { id: 'score', x: 940, y: 85, width: 230, height: 165 },
    ];

    const res = computeSmartEdgePath({
      sourceX: 270,
      sourceY: 235,
      targetX: 940,
      targetY: 180,
      sourceId: 'rfd4',
      targetId: 'score',
      nodes,
      padding: 18,
    });

    expect(res).not.toBeNull();
    // Should wrap BELOW mpnn (bottom=235) and fold (bottom=225) because bottom detour is much shorter than top
    const ys = res!.points.map((p) => p.y);
    const maxY = Math.max(...ys);
    expect(maxY).toBeGreaterThan(235 + 18);

    // Zero intersections with ProteinMPNN or LocalFold (even with 18px padding)
    const obstacleBoxes = nodes
      .filter((n) => n.id === 'mpnn' || n.id === 'fold')
      .map((n) => ({
        left: n.x - 18,
        right: n.x + n.width + 18,
        top: n.y - 18,
        bottom: n.y + n.height + 18,
      }));
    expect(countPolylineHits(res!.points, obstacleBoxes)).toBe(0);
    expect(res!.path).toMatch(/^M /);
    expect(res!.path).toContain(' Q ');
  });

  it('chooses the top lane when the top detour is shorter', () => {
    const nodes: ObstacleRect[] = [
      { id: 'src', x: 40, y: 60, width: 220, height: 140 },
      { id: 'mid', x: 340, y: 80, width: 220, height: 180 },
      { id: 'dst', x: 640, y: 60, width: 220, height: 140 },
    ];
    const res = computeSmartEdgePath({
      sourceX: 260,
      sourceY: 100,
      targetX: 640,
      targetY: 100,
      sourceId: 'src',
      targetId: 'dst',
      nodes,
      padding: 18,
    });
    expect(res).not.toBeNull();
    const minY = Math.min(...res!.points.map((p) => p.y));
    expect(minY).toBeLessThan(80 - 18);
  });

  it('threads a clear horizontal corridor between stacked intermediate rows', () => {
    const nodes: ObstacleRect[] = [
      { id: 'src', x: 40, y: 170, width: 220, height: 140 },
      { id: 'topMid', x: 360, y: 40, width: 220, height: 130 },
      { id: 'botMid', x: 360, y: 280, width: 220, height: 130 },
      { id: 'dst', x: 680, y: 170, width: 220, height: 140 },
    ];
    // Corridor between y=170+18=188 and y=280-18=262 (midpoint 225)
    // Source at y=160 hits topMid (bottom=188), target at y=160 hits topMid -> corridor at y=225 is much shorter than going above y=40 or below y=410!
    const res = computeSmartEdgePath({
      sourceX: 260,
      sourceY: 160,
      targetX: 680,
      targetY: 160,
      sourceId: 'src',
      targetId: 'dst',
      nodes,
      padding: 18,
    });
    expect(res).not.toBeNull();
    const laneY = res!.points[2].y;
    expect(laneY).toBeGreaterThan(188);
    expect(laneY).toBeLessThan(262);
  });

  it('uses a clean 2-bend shifted Z-path when a single obstacle only blocks the midpoint', () => {
    const nodes: ObstacleRect[] = [
      { id: 'src', x: 40, y: 60, width: 220, height: 120 },
      // Obstacle sits around the midpoint x=450, y=140..280, blocking midX=450 vertical leg,
      // while x=290 (just right of src) is completely clear vertically from y=100 to y=300!
      { id: 'obs', x: 400, y: 140, width: 200, height: 140 },
      { id: 'dst', x: 660, y: 240, width: 220, height: 120 },
    ];
    const res = computeSmartEdgePath({
      sourceX: 260,
      sourceY: 100,
      targetX: 660,
      targetY: 300,
      sourceId: 'src',
      targetId: 'dst',
      nodes,
      padding: 18,
    });
    expect(res).not.toBeNull();
    // A 3-segment Z-path has 4 points (only 2 bends)
    expect(res!.points.length).toBe(4);
    expect(res!.points[1].x).toBeLessThan(400 - 18);
  });

  it('wraps cleanly around source and target cards on backward edges (target left of source)', () => {
    const nodes: ObstacleRect[] = [
      { id: 'src', x: 420, y: 120, width: 220, height: 140 },
      { id: 'dst', x: 80, y: 120, width: 220, height: 140 },
    ];
    const res = computeSmartEdgePath({
      sourceX: 640,
      sourceY: 180,
      targetX: 80,
      targetY: 180,
      sourceId: 'src',
      targetId: 'dst',
      nodes,
      padding: 18,
    });
    expect(res).not.toBeNull();
    // Middle horizontal segment must be completely above (y < 102) or below (y > 278) both cards
    const laneY = res!.points[2].y;
    expect(laneY < 120 - 18 || laneY > 260 + 18).toBe(true);
  });

  it('cleans collinear points and formats valid rounded SVG paths', () => {
    const cleaned = cleanPolyline([
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
    ]);
    expect(cleaned).toEqual([
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
    ]);
    const built = pointsToRoundedPath(cleaned, 12);
    expect(built.path).toBe('M 0.0 0.0 L 88.0 0.0 Q 100.0 0.0 100.0 12.0 L 100.0 100.0');
  });

  it('Issue #104: computes a non-degenerate 2D hit rect for straight horizontal & vertical edges (Safari SVG hit-test)', () => {
    const horiz = computeStraightEdgeHitRect({
      sourceX: 260,
      sourceY: 170,
      targetX: 380,
      targetY: 170,
      points: null,
      interactionWidth: 22,
    });
    expect(horiz).not.toBeNull();
    expect(horiz!.width).toBe(112); // 120px span minus 4px inset on each end
    expect(horiz!.height).toBe(22);
    expect(horiz!.x).toBe(264);
    expect(horiz!.y).toBe(159);

    const vert = computeStraightEdgeHitRect({
      sourceX: 300,
      sourceY: 100,
      targetX: 300,
      targetY: 220,
      points: null,
      interactionWidth: 22,
    });
    expect(vert).not.toBeNull();
    expect(vert!.width).toBe(22);
    expect(vert!.height).toBe(112);

    // Bent Z-path or multi-segment detour returns null (already has non-zero 2D bbox)
    expect(
      computeStraightEdgeHitRect({
        sourceX: 260,
        sourceY: 170,
        targetX: 380,
        targetY: 230,
        points: null,
      }),
    ).toBeNull();
    expect(
      computeStraightEdgeHitRect({
        sourceX: 260,
        sourceY: 170,
        targetX: 80,
        targetY: 170,
        points: [
          { x: 260, y: 170 },
          { x: 284, y: 170 },
          { x: 284, y: 300 },
          { x: 56, y: 300 },
          { x: 56, y: 170 },
          { x: 80, y: 170 },
        ],
      }),
    ).toBeNull();
  });

  it('Issue #104: SmartEdge provides explicit stroke hit-target & WorkflowCanvas disables Shift selection-box hijack', () => {
    const edgeSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/SmartEdge.svelte'), 'utf8');
    expect(edgeSrc).toContain('smart-edge-hit-rect');
    expect(edgeSrc).toContain('pointer-events="all"');
    expect(edgeSrc).toContain('.svelte-flow__edge-interaction');
    expect(edgeSrc).toContain('pointer-events: stroke');

    const canvasSrc = readFileSync(resolve(process.cwd(), 'src/lib/ui/WorkflowCanvas.svelte'), 'utf8');
    expect(canvasSrc).toContain('panOnDrag={true}');
    expect(canvasSrc).toContain('selectionOnDrag={false}');
    expect(canvasSrc).toContain('selectionKey={[]}');
  });
});
