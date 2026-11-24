import { OgCanvas, hexToRgba, type EdgeStructurePreview } from './raster';
import { CATEGORY_COLORS, type DecodedOgWorkflow, type OgNode } from './og-graph';

export const OG_WIDTH = 1200;
export const OG_HEIGHT = 630;

interface LayoutCard {
  node: OgNode;
  x: number;
  y: number;
  w: number;
  h: number;
}

function layoutNodesInRegion(
  workflow: DecodedOgWorkflow,
  rx: number,
  ry: number,
  rw: number,
  rh: number,
): Map<string, LayoutCard> {
  const cards = new Map<string, LayoutCard>();
  const nodes = workflow.nodes;
  if (nodes.length === 0) return cards;

  // Compute topological columns so workflows read cleanly left -> right
  const inDeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const n of nodes) {
    inDeg.set(n.id, 0);
    adj.set(n.id, []);
  }
  for (const e of workflow.edges) {
    if (adj.has(e.fromNode) && inDeg.has(e.toNode)) {
      adj.get(e.fromNode)!.push(e.toNode);
      inDeg.set(e.toNode, (inDeg.get(e.toNode) ?? 0) + 1);
    }
  }
  const layerByNode = new Map<string, number>();
  const queue = nodes.filter((n) => (inDeg.get(n.id) ?? 0) === 0).map((n) => n.id);
  for (const id of queue) layerByNode.set(id, 0);
  while (queue.length > 0) {
    const u = queue.shift()!;
    const lu = layerByNode.get(u) ?? 0;
    for (const v of adj.get(u) ?? []) {
      const nextL = Math.max(layerByNode.get(v) ?? 0, lu + 1);
      layerByNode.set(v, nextL);
      inDeg.set(v, (inDeg.get(v) ?? 1) - 1);
      if ((inDeg.get(v) ?? 0) <= 0) queue.push(v);
    }
  }
  for (let i = 0; i < nodes.length; i++) {
    if (!layerByNode.has(nodes[i].id)) layerByNode.set(nodes[i].id, i);
  }

  const cols = new Map<number, OgNode[]>();
  for (const n of nodes) {
    const col = layerByNode.get(n.id) ?? 0;
    const arr = cols.get(col) ?? [];
    arr.push(n);
    cols.set(col, arr);
  }
  const sortedColKeys = [...cols.keys()].sort((a, b) => a - b);
  const numCols = sortedColKeys.length;
  const maxRows = Math.max(...[...cols.values()].map((c) => c.length));

  const padX = 28;
  const padY = 26;
  const availW = rw - padX * 2;
  const availH = rh - padY * 2;

  // For linear 4-block pipelines inside a split side-by-side panel (rw < 760), use a crisp 2x2 grid
  if (numCols >= 3 && maxRows === 1 && availW < numCols * 185) {
    const gridCols = 2;
    const gridRows = Math.ceil(numCols / gridCols);
    const gapX = 44;
    const gapY = 44;
    const cardW = Math.min(280, Math.floor((availW - gapX * (gridCols - 1)) / gridCols));
    const cardH = Math.min(118, Math.floor((availH - gapY * (gridRows - 1)) / gridRows));
    const totalW = gridCols * cardW + (gridCols - 1) * gapX;
    const totalH = gridRows * cardH + (gridRows - 1) * gapY;
    const startX = rx + Math.round((rw - totalW) / 2);
    const startY = ry + Math.round((rh - totalH) / 2);
    for (let idx = 0; idx < numCols; idx++) {
      const node = cols.get(sortedColKeys[idx])![0];
      const gr = Math.floor(idx / gridCols);
      const gc = idx % gridCols;
      cards.set(node.id, {
        node,
        x: startX + gc * (cardW + gapX),
        y: startY + gr * (cardH + gapY),
        w: cardW,
        h: cardH,
      });
    }
    return cards;
  }

  const gapX = numCols <= 2 ? 42 : numCols === 3 ? 34 : 28;
  const gapY = 26;

  let cardW = Math.min(260, Math.floor((availW - gapX * (numCols - 1)) / numCols));
  if (nodes.length === 1) cardW = Math.min(340, availW - 24);
  cardW = Math.max(150, cardW);
  const cardH = Math.min(
    nodes.length === 1 ? 136 : 114,
    Math.max(88, Math.floor((availH - gapY * (maxRows - 1)) / maxRows)),
  );

  const totalW = numCols * cardW + (numCols - 1) * gapX;
  const startX = rx + Math.round((rw - totalW) / 2);

  for (let c = 0; c < numCols; c++) {
    const colNodes = cols.get(sortedColKeys[c])!;
    const totalColH = colNodes.length * cardH + (colNodes.length - 1) * gapY;
    const startY = ry + Math.round((rh - totalColH) / 2);
    for (let r = 0; r < colNodes.length; r++) {
      const node = colNodes[r];
      cards.set(node.id, {
        node,
        x: startX + c * (cardW + gapX),
        y: startY + r * (cardH + gapY),
        w: cardW,
        h: cardH,
      });
    }
  }
  return cards;
}

function drawWorkflowNodeCard(canvas: OgCanvas, card: LayoutCard): void {
  const { node, x, y, w, h } = card;
  const style = CATEGORY_COLORS[node.category] ?? CATEGORY_COLORS.evaluation;
  const radius = 14;

  canvas.drawCardShadow(x, y, w, h, radius, 12);
  canvas.fillRoundedRect(x, y, w, h, radius, hexToRgba('#ffffff'));
  canvas.fillTopRoundedBar(x, y, w, 5, h, radius, hexToRgba(style.accent));
  canvas.strokeRoundedRect(x, y, w, h, radius, 1.5, hexToRgba('#cbd5e1'));

  // Category tag
  canvas.drawText(style.label, x + 16, y + 27, 'reg12', hexToRgba(style.accent));

  // Node title
  canvas.drawEllipsisText(node.label, x + 16, y + 53, w - 32, 'semi18', hexToRgba('#0f172a'));

  // Param pill
  const chipY = y + h - 38;
  const chipH = 26;
  canvas.fillRoundedRect(x + 14, chipY, w - 28, chipH, 7, hexToRgba('#f1f5f9'));
  canvas.strokeRoundedRect(x + 14, chipY, w - 28, chipH, 7, 1, hexToRgba('#e2e8f0'));
  canvas.drawEllipsisText(
    node.paramChip,
    x + 22,
    chipY + 18,
    w - 44,
    'reg12',
    hexToRgba('#475569'),
  );
}

/**
 * Render a 1200x630 social-media Open Graph PNG for a decoded denovo.run workflow,
 * combining the workflow DAG canvas and (when available) the 3D protein structure.
 */
export async function renderWorkflowOgPng(
  workflow: DecodedOgWorkflow,
  structure: EdgeStructurePreview | null = null,
): Promise<Uint8Array> {
  const canvas = new OgCanvas(OG_WIDTH, OG_HEIGHT, '#f8fafc');
  canvas.fillLinearGradientRect(0, 0, OG_WIDTH, OG_HEIGHT, '#f8fafc', '#eef2f9');
  canvas.drawDotGrid(0, 82, OG_WIDTH, OG_HEIGHT - 134, 24, '#cbd5e1', 0.55);

  // Header bar
  canvas.fillRoundedRect(0, 0, OG_WIDTH, 82, 0, hexToRgba('#ffffff', 246));
  canvas.fillRoundedRect(0, 81, OG_WIDTH, 1, 0, hexToRgba('#e2e8f0'));

  canvas.drawBrandLogo(32, 20, 42);
  canvas.drawText('denovo', 86, 51, 'bold28', hexToRgba('#2563eb'));
  const wDenovo = canvas.measureText('denovo', 'bold28');
  canvas.drawText('.run', 86 + wDenovo, 51, 'bold28', hexToRgba('#0f172a'));

  // Right-aligned header pill
  const pillText = 'Browser-Native WebGPU · 100% Local';
  const pillW = canvas.measureText(pillText, 'med14') + 30;
  const pillX = OG_WIDTH - 32 - pillW;
  canvas.fillRoundedRect(pillX, 24, pillW, 34, 17, hexToRgba('#eff6ff'));
  canvas.strokeRoundedRect(pillX, 24, pillW, 34, 17, 1.2, hexToRgba('#bfdbfe'));
  canvas.drawText(pillText, pillX + 15, 46, 'med14', hexToRgba('#1d4ed8'));

  // Main body area: y = 96..568 (height 472)
  const bodyY = 98;
  const bodyH = 468;

  if (structure) {
    const isSingle = workflow.nodes.length <= 1;
    const dagW = isSingle ? 470 : 695;
    const structX = 32 + dagW + 22;
    const structW = OG_WIDTH - structX - 32;

    // Left: Workflow canvas panel
    canvas.drawCardShadow(32, bodyY, dagW, bodyH, 18, 10);
    canvas.fillRoundedRect(32, bodyY, dagW, bodyH, 18, hexToRgba('#f8fafc', 238));
    canvas.drawDotGrid(36, bodyY + 4, dagW - 8, bodyH - 8, 22, '#cbd5e1', 0.55);
    canvas.strokeRoundedRect(32, bodyY, dagW, bodyH, 18, 1.5, hexToRgba('#e2e8f0'));

    canvas.drawText('WORKFLOW PIPELINE', 52, bodyY + 30, 'reg12', hexToRgba('#64748b'));

    const cards = layoutNodesInRegion(workflow, 32, bodyY + 26, dagW, bodyH - 36);
    drawEdges(canvas, workflow, cards);
    for (const card of cards.values()) {
      drawWorkflowNodeCard(canvas, card);
    }

    // Right: Hero 3D structure viewport
    canvas.drawCardShadow(structX, bodyY, structW, bodyH, 18, 10);
    canvas.fillRoundedRect(structX, bodyY, structW, bodyH, 18, hexToRgba('#0f172a'));
    canvas.strokeRoundedRect(structX, bodyY, structW, bodyH, 18, 1.5, hexToRgba('#1e293b'));

    // Subtle radial-like vignette corners & structure label pill
    canvas.renderProtein3D(structure, structX + 8, bodyY + 42, structW - 16, bodyH - 88);

    // Top-left 3D badge
    const badgeW = Math.min(structW - 36, canvas.measureText(structure.title, 'med14') + 28);
    canvas.fillRoundedRect(structX + 18, bodyY + 16, badgeW, 30, 8, hexToRgba('#1e293b', 230));
    canvas.strokeRoundedRect(structX + 18, bodyY + 16, badgeW, 30, 8, 1, hexToRgba('#334155'));
    canvas.fillCircle(structX + 31, bodyY + 31, 4, hexToRgba('#34d399'));
    canvas.drawEllipsisText(
      structure.title,
      structX + 41,
      bodyY + 36,
      badgeW - 28,
      'med14',
      hexToRgba('#f8fafc'),
    );

    // Bottom 3D metadata bar
    canvas.drawEllipsisText(
      structure.subtitle,
      structX + 20,
      bodyY + bodyH - 18,
      structW - 40,
      'reg12',
      hexToRgba('#94a3b8'),
    );
  } else {
    // Full-width crisp DAG canvas for pure de novo / sequence pipelines
    const fullW = OG_WIDTH - 64;
    canvas.drawCardShadow(32, bodyY, fullW, bodyH, 18, 10);
    canvas.fillRoundedRect(32, bodyY, fullW, bodyH, 18, hexToRgba('#f8fafc', 238));
    canvas.drawDotGrid(36, bodyY + 4, fullW - 8, bodyH - 8, 22, '#cbd5e1', 0.55);
    canvas.strokeRoundedRect(32, bodyY, fullW, bodyH, 18, 1.5, hexToRgba('#e2e8f0'));

    canvas.drawText('WORKFLOW PIPELINE', 56, bodyY + 32, 'reg12', hexToRgba('#64748b'));

    const cards = layoutNodesInRegion(workflow, 32, bodyY + 26, fullW, bodyH - 36);
    drawEdges(canvas, workflow, cards);
    for (const card of cards.values()) {
      drawWorkflowNodeCard(canvas, card);
    }
  }

  // Footer strip
  const cleanTitle = workflow.title.replace(/\s+—\s+denovo\.run$/, '');
  canvas.drawEllipsisText(cleanTitle, 34, OG_HEIGHT - 22, OG_WIDTH - 240, 'med14', hexToRgba('#0f172a'));
  const blockCountText = `${workflow.nodes.length} block${workflow.nodes.length === 1 ? '' : 's'} · Open in browser →`;
  const rightW = canvas.measureText(blockCountText, 'med14');
  canvas.drawText(blockCountText, OG_WIDTH - 34 - rightW, OG_HEIGHT - 22, 'med14', hexToRgba('#2563eb'));

  return canvas.encodePng();
}

function drawEdges(
  canvas: OgCanvas,
  workflow: DecodedOgWorkflow,
  cards: Map<string, LayoutCard>,
): void {
  const edgeColors: Record<string, string> = {
    structure: '#3b82f6',
    sequences: '#8b5cf6',
    folds: '#10b981',
    metrics: '#f59e0b',
  };
  for (const e of workflow.edges) {
    const src = cards.get(e.fromNode);
    const dst = cards.get(e.toNode);
    if (!src || !dst) continue;
    const colorHex = edgeColors[e.fromPort] ?? '#64748b';
    // Handle same-row left->right edges vs wrapped next-row edges cleanly
    if (dst.x >= src.x + src.w - 8) {
      const x0 = src.x + src.w;
      const y0 = src.y + src.h * 0.5;
      const x1 = dst.x;
      const y1 = dst.y + dst.h * 0.5;
      const dx = Math.max(36, Math.abs(x1 - x0) * 0.52);
      canvas.strokeCubicBezier(
        x0,
        y0,
        x0 + dx,
        y0,
        x1 - dx,
        y1,
        x1 - 3,
        y1,
        2.2,
        hexToRgba(colorHex),
        true,
      );
      canvas.fillCircle(x0, y0, 4.2, hexToRgba(colorHex));
      canvas.fillCircle(x1, y1, 4.2, hexToRgba(colorHex));
    } else {
      const x0 = src.x + src.w * 0.5;
      const y0 = src.y + src.h;
      const x1 = dst.x + dst.w * 0.5;
      const y1 = dst.y;
      const dy = Math.max(24, Math.abs(y1 - y0) * 0.55);
      canvas.strokeCubicBezier(
        x0,
        y0,
        x0,
        y0 + dy,
        x1,
        y1 - dy,
        x1,
        y1 - 3,
        2.2,
        hexToRgba(colorHex),
        true,
      );
      canvas.fillCircle(x0, y0, 4.2, hexToRgba(colorHex));
      canvas.fillCircle(x1, y1, 4.2, hexToRgba(colorHex));
    }
  }
}

/** Generate the default static 1200x630 Open Graph PNG for bare https://denovo.run links. */
export async function renderDefaultOgPng(): Promise<Uint8Array> {
  const demoWorkflow: DecodedOgWorkflow = {
    title: 'Genie 3 → Cα-ProteinMPNN → AF2 Model 1 → scRMSD / scTM — denovo.run',
    description: 'Browser-native WebGPU protein design workbench.',
    nodes: [
      {
        id: 'gen',
        adapterId: 'genie3',
        label: 'Genie 3',
        category: 'generation',
        paramChip: '80 aa · 100 steps',
        params: {},
      },
      {
        id: 'mpnn',
        adapterId: 'web-protein-mpnn',
        label: 'Cα-ProteinMPNN',
        category: 'sequence',
        paramChip: '8 seqs · T=0.1',
        params: {},
      },
      {
        id: 'fold',
        adapterId: 'localfold-af2',
        label: 'AF2 Model 1',
        category: 'prediction',
        paramChip: '1 rec · Auto (≤8)',
        params: {},
      },
      {
        id: 'score',
        adapterId: 'designability-scorer',
        label: 'scRMSD / scTM',
        category: 'evaluation',
        paramChip: 'scRMSD < 2.0 Å · pLDDT ≥ 70',
        params: {},
      },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'structure', toNode: 'mpnn', toPort: 'structure' },
      { fromNode: 'mpnn', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
    ],
    structureTarget: null,
  };
  return renderWorkflowOgPng(demoWorkflow, null);
}
