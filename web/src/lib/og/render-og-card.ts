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
  canvas.drawDotGrid(0, 76, OG_WIDTH, OG_HEIGHT - 76, 24, '#cbd5e1', 0.55);

  // Clean minimal header bar (brand mark only — leaves top-right clear for platform card overlays)
  canvas.fillRoundedRect(0, 0, OG_WIDTH, 76, 0, hexToRgba('#ffffff', 246));
  canvas.fillRoundedRect(0, 75, OG_WIDTH, 1, 0, hexToRgba('#e2e8f0'));

  canvas.drawBrandLogo(32, 17, 42);
  canvas.drawText('denovo', 86, 48, 'bold28', hexToRgba('#2563eb'));
  const wDenovo = canvas.measureText('denovo', 'bold28');
  canvas.drawText('.run', 86 + wDenovo, 48, 'bold28', hexToRgba('#0f172a'));

  // Main body area fills cleanly to bottom margin (social platforms render title/description below)
  const bodyY = 96;
  const bodyH = OG_HEIGHT - bodyY - 28;

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

// Baked 10x-quantized Streptavidin–Biotin (RCSB 1STP Chain A + BTN) trace for zero-fetch default OG card.
const HERO_1STP_CA: ReadonlyArray<readonly [number, number, number]> = [
  [237,49,111],[258,73,91],[287,49,86],[263,20,82],[239,36,57],[265,52,36],[276,31,7],[264,14,-25],
  [235,-10,-23],[217,-33,-47],[181,-46,-46],[166,-79,-58],[162,-64,-94],[198,-52,-98],[191,-16,-92],
  [219,5,-77],[214,29,-48],[243,51,-39],[238,70,-6],[255,93,19],[238,97,53],[245,132,67],[250,138,104],
  [218,159,104],[196,130,92],[198,139,55],[201,113,27],[215,124,-7],[228,102,-36],[220,86,-69],
  [196,59,-80],[197,37,-112],[169,13,-120],[174,-20,-138],[138,-18,-150],[110,5,-161],[110,38,-180],
  [144,47,-168],[181,50,-179],[213,61,-162],[214,78,-128],[187,99,-111],[190,123,-81],[175,111,-48],
  [168,130,-16],[156,115,18],[157,116,56],[166,90,83],[163,87,120],[197,100,131],[193,89,168],
  [179,54,166],[167,36,197],[194,16,215],[173,-16,219],[186,-30,186],[155,-19,165],[161,-9,129],
  [165,28,118],[135,43,100],[143,61,67],[128,71,34],[138,86,0],[123,99,-32],[133,99,-69],[115,120,-95],
  [121,100,-127],[136,128,-148],[173,132,-158],[191,145,-189],[183,114,-209],[146,111,-201],
  [114,119,-182],[90,92,-172],[67,118,-154],[66,97,-123],[80,96,-87],[88,67,-64],[90,69,-26],
  [107,41,-6],[98,38,31],[117,12,52],[142,6,79],[180,-1,80],[191,-32,99],[228,-34,108],[250,-64,115],
  [268,-92,97],[246,-90,67],[232,-54,68],[200,-48,49],[175,-21,40],[139,-34,39],[115,-11,21],
  [80,-7,6],[75,18,-22],[48,38,-39],[53,47,-76],[30,75,-88],[25,86,-124],[12,120,-134],[-22,111,-148],
  [-54,128,-161],[-83,129,-135],[-101,102,-153],[-72,78,-147],[-61,88,-112],[-77,56,-98],
  [-48,37,-115],[-19,61,-109],[0,43,-82],[19,10,-80],[31,-7,-47],[59,-32,-41],[89,-39,-18],
  [126,-39,-24],[154,-54,-2],[189,-38,-5],[222,-54,6],[258,-43,1],[257,-76,-18],
];
const HERO_1STP_BTN: ReadonlyArray<readonly [number, number, number]> = [
  [91,46,-145],[81,54,-142],[87,35,-151],[103,44,-137],[102,32,-127],[115,30,-120],[116,16,-114],
  [109,11,-102],[109,21,-87],[103,6,-78],[111,-5,-83],[124,-6,-76],[135,-4,-85],[147,-4,-82],
  [130,-2,-97],[115,-3,-97],
];

/** Generate the default static 1200x630 Open Graph PNG for bare https://denovo.run links. */
export async function renderDefaultOgPng(): Promise<Uint8Array> {
  const demoWorkflow: DecodedOgWorkflow = {
    title: 'Interactive WebGPU Molecular Design Workspace — denovo.run',
    description:
      'Have a molecular design idea you want to quickly prototype in your browser? Interactive browser-native WebGPU workspace for structure, design, folding & docking.',
    nodes: [
      {
        id: 'gen',
        adapterId: 'rfd4-proteina',
        label: 'Structure Generation',
        category: 'generation',
        paramChip: 'RFD4 · RFdiffusion · BoltzGen · +',
        params: {},
      },
      {
        id: 'seq',
        adapterId: 'web-protein-mpnn',
        label: 'Inverse Folding',
        category: 'sequence',
        paramChip: 'ProteinMPNN · LigandMPNN · +',
        params: {},
      },
      {
        id: 'fold',
        adapterId: 'localfold-af2',
        label: 'Structure Prediction',
        category: 'prediction',
        paramChip: 'AF2 · Boltz-2 · RF3 · ESMFold2',
        params: {},
      },
      {
        id: 'score',
        adapterId: 'designability-scorer',
        label: 'Scoring & Docking',
        category: 'evaluation',
        paramChip: 'scRMSD · GNINA · ΔΔG · ESM-C',
        params: {},
      },
    ],
    edges: [
      { fromNode: 'gen', fromPort: 'structure', toNode: 'seq', toPort: 'structure' },
      { fromNode: 'seq', fromPort: 'sequences', toNode: 'fold', toPort: 'sequences' },
      { fromNode: 'fold', fromPort: 'folds', toNode: 'score', toPort: 'folds' },
      { fromNode: 'gen', fromPort: 'structure', toNode: 'score', toPort: 'structure' },
    ],
    structureTarget: null,
  };

  const heroStructure: EdgeStructurePreview = {
    title: '3D Complex · Protein + Ligand',
    subtitle: 'Streptavidin–Biotin (1STP) · L=121 aa · BTN',
    chains: [
      {
        id: 'A',
        points: HERO_1STP_CA.map(([x, y, z]) => [x / 10, y / 10, z / 10] as const),
      },
    ],
    ligands: [
      {
        name: 'BTN',
        chain: 'A',
        atoms: HERO_1STP_BTN.map(([x, y, z]) => [x / 10, y / 10, z / 10] as const),
      },
    ],
    residueCount: HERO_1STP_CA.length,
    isPlddt: false,
  };

  return renderWorkflowOgPng(demoWorkflow, heroStructure);
}
