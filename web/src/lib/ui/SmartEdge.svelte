<script lang="ts">
  import {
    BaseEdge,
    getSmoothStepPath,
    useNodes,
    type EdgeProps,
  } from '@xyflow/svelte';
  import { computeSmartEdgePath, type ObstacleRect } from './edge-routing';

  let {
    id,
    source,
    target,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    markerStart,
    markerEnd,
    style,
    interactionWidth,
    label,
    labelStyle,
    data,
  }: EdgeProps = $props();

  const nodes = useNodes();

  let routed = $derived.by(() => {
    const rects: ObstacleRect[] = nodes.current.map((n) => ({
      id: n.id,
      x: n.position.x,
      y: n.position.y,
      width: n.measured?.width ?? n.width ?? 230,
      height: n.measured?.height ?? n.height ?? 160,
    }));

    const laneStagger =
      typeof data?.laneStagger === 'number' ? data.laneStagger : 0;

    const smart = computeSmartEdgePath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourceId: source,
      targetId: target,
      nodes: rects,
      stub: 24,
      padding: 18,
      borderRadius: 12,
      laneStagger,
    });

    if (smart) {
      return { path: smart.path, labelX: smart.labelX, labelY: smart.labelY };
    }

    const [path, labelX, labelY] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 14,
      offset: 26,
    });
    return { path, labelX, labelY };
  });
</script>

<BaseEdge
  {id}
  path={routed.path}
  labelX={routed.labelX}
  labelY={routed.labelY}
  {label}
  {labelStyle}
  {markerStart}
  {markerEnd}
  {interactionWidth}
  {style}
/>
