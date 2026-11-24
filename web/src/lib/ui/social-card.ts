import { ADAPTER_REGISTRY } from '$lib/engine/registry';
import type { CanonicalArtifact, WorkflowGraph, WorkflowNode } from '$lib/engine/types';
import { formatNodeParamSummary } from './node-display-helpers';

export type SocialCardMode = 'both' | 'structure' | 'canvas';

export interface SocialCardOptions {
  graph: WorkflowGraph;
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>;
  structurePngDataUrl?: string | null;
  structureLabel?: string;
  structureMeta?: string;
  mode: SocialCardMode;
  theme?: 'light' | 'dark';
}

const CARD_W = 1200;
const CARD_H = 630;

const STAGE_COLORS: Record<string, { accent: string; label: string }> = {
  source: { accent: '#0d9488', label: 'INPUT' },
  condition: { accent: '#0d9488', label: 'CONDITION' },
  generate: { accent: '#2563eb', label: 'GENERATION' },
  'inverse-fold': { accent: '#7c3aed', label: 'SEQUENCE DESIGN' },
  fold: { accent: '#059669', label: 'PREDICTION' },
  score: { accent: '#d97706', label: 'EVALUATION' },
};

function loadImage(dataUrl: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function drawEllipsis(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
): void {
  if (ctx.measureText(text).width <= maxWidth) {
    ctx.fillText(text, x, y);
    return;
  }
  let out = text;
  while (out.length > 1 && ctx.measureText(out + '…').width > maxWidth) {
    out = out.slice(0, -1);
  }
  ctx.fillText(out.trimEnd() + '…', x, y);
}

function summarizeNodeResultBadge(
  _node: WorkflowNode,
  outputs: Record<string, CanonicalArtifact> | undefined,
): string | null {
  if (!outputs) return null;
  const metrics = outputs.metrics;
  if (metrics && metrics.kind === 'DesignabilityMetrics') {
    const best = metrics.best ?? metrics.rows?.[0];
    if (best) {
      const parts: string[] = [];
      if (typeof best.scRmsd === 'number' && Number.isFinite(best.scRmsd)) {
        parts.push(`scRMSD ${best.scRmsd.toFixed(2)} Å`);
      }
      if (typeof best.meanPlddt === 'number' && Number.isFinite(best.meanPlddt)) {
        parts.push(`pLDDT ${best.meanPlddt.toFixed(1)}`);
      }
      if (parts.length > 0) return parts.join(' · ');
    }
  }
  const folds = outputs.folds;
  if (folds && folds.kind === 'FoldedStructure' && folds.folds.length > 0) {
    const plddts = folds.folds
      .map((s) => s.meanPlddt)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const best = plddts.length > 0 ? Math.max(...plddts) : null;
    return best !== null
      ? `${folds.folds.length} fold${folds.folds.length === 1 ? '' : 's'} · pLDDT ${best.toFixed(1)}`
      : `${folds.folds.length} fold${folds.folds.length === 1 ? '' : 's'}`;
  }
  const seqs = outputs.sequences;
  if (seqs && seqs.kind === 'SequenceSet' && seqs.candidates.length > 0) {
    return `${seqs.candidates.length} seq${seqs.candidates.length === 1 ? '' : 's'}`;
  }
  const struct = outputs.structure;
  if (struct && struct.kind === 'StructureArtifact') {
    return `L=${struct.length} aa`;
  }
  return null;
}

function drawBrandMark(ctx: CanvasRenderingContext2D, x: number, y: number, size: number): void {
  const s = size / 32;
  ctx.save();
  roundRectPath(ctx, x, y, size, size, 7 * s);
  ctx.fillStyle = '#0f172a';
  ctx.fill();

  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 2.8 * s;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x + 8 * s, y + 22 * s);
  ctx.bezierCurveTo(x + 11 * s, y + 10 * s, x + 15 * s, y + 10 * s, x + 16 * s, y + 16 * s);
  ctx.bezierCurveTo(x + 17 * s, y + 22 * s, x + 21 * s, y + 22 * s, x + 24 * s, y + 10 * s);
  ctx.stroke();

  for (const [cx, cy, col] of [
    [8, 22, '#38bdf8'],
    [16, 16, '#a78bfa'],
    [24, 10, '#34d399'],
  ] as const) {
    ctx.beginPath();
    ctx.arc(x + cx * s, y + cy * s, 2.7 * s, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
  }
  ctx.restore();
}

function drawDagPanel(
  ctx: CanvasRenderingContext2D,
  graph: WorkflowGraph,
  outputsByNode: Record<string, Record<string, CanonicalArtifact>>,
  rx: number,
  ry: number,
  rw: number,
  rh: number,
  isDark: boolean,
): void {
  ctx.save();
  ctx.shadowColor = 'rgba(15, 23, 42, 0.10)';
  ctx.shadowBlur = 18;
  ctx.shadowOffsetY = 5;
  roundRectPath(ctx, rx, ry, rw, rh, 18);
  ctx.fillStyle = isDark ? '#111827' : '#f8fafc';
  ctx.fill();
  ctx.restore();

  ctx.save();
  roundRectPath(ctx, rx, ry, rw, rh, 18);
  ctx.clip();

  // Dot grid
  ctx.fillStyle = isDark ? 'rgba(148, 163, 184, 0.16)' : 'rgba(148, 163, 184, 0.36)';
  for (let gy = ry + 16; gy < ry + rh; gy += 22) {
    for (let gx = rx + 16; gx < rx + rw; gx += 22) {
      ctx.beginPath();
      ctx.arc(gx, gy, 1.15, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
  ctx.font = '700 11px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
  ctx.fillText('WORKFLOW PIPELINE', rx + 20, ry + 28);

  const nodes = graph.nodes;
  if (nodes.length > 0) {
    const inDeg = new Map<string, number>();
    const adj = new Map<string, string[]>();
    for (const n of nodes) {
      inDeg.set(n.id, 0);
      adj.set(n.id, []);
    }
    for (const e of graph.edges) {
      if (adj.has(e.fromNode) && inDeg.has(e.toNode)) {
        adj.get(e.fromNode)!.push(e.toNode);
        inDeg.set(e.toNode, (inDeg.get(e.toNode) ?? 0) + 1);
      }
    }
    const layer = new Map<string, number>();
    const q = nodes.filter((n) => (inDeg.get(n.id) ?? 0) === 0).map((n) => n.id);
    for (const id of q) layer.set(id, 0);
    while (q.length > 0) {
      const u = q.shift()!;
      const lu = layer.get(u) ?? 0;
      for (const v of adj.get(u) ?? []) {
        layer.set(v, Math.max(layer.get(v) ?? 0, lu + 1));
        inDeg.set(v, (inDeg.get(v) ?? 1) - 1);
        if ((inDeg.get(v) ?? 0) <= 0) q.push(v);
      }
    }
    for (let i = 0; i < nodes.length; i++) {
      if (!layer.has(nodes[i].id)) layer.set(nodes[i].id, i);
    }

    const cols = new Map<number, WorkflowNode[]>();
    for (const n of nodes) {
      const c = layer.get(n.id) ?? 0;
      const arr = cols.get(c) ?? [];
      arr.push(n);
      cols.set(c, arr);
    }
    const colKeys = [...cols.keys()].sort((a, b) => a - b);
    const numCols = colKeys.length;
    const maxRows = Math.max(...[...cols.values()].map((c) => c.length));

    const gapX = numCols <= 2 ? 40 : 28;
    const gapY = 24;
    const padX = 24;
    const padY = 44;
    const availW = rw - padX * 2;
    const availH = rh - padY - 22;

    let cardW = Math.min(250, Math.floor((availW - gapX * (numCols - 1)) / numCols));
    if (nodes.length === 1) cardW = Math.min(340, availW - 20);
    cardW = Math.max(145, cardW);
    const cardH = Math.min(118, Math.floor((availH - gapY * (maxRows - 1)) / maxRows));
    const totalW = numCols * cardW + (numCols - 1) * gapX;
    const startX = rx + Math.round((rw - totalW) / 2);

    const rects = new Map<string, { x: number; y: number; w: number; h: number; node: WorkflowNode }>();
    for (let ci = 0; ci < numCols; ci++) {
      const colNodes = cols.get(colKeys[ci])!;
      const totalH = colNodes.length * cardH + (colNodes.length - 1) * gapY;
      const startY = ry + 36 + Math.round((rh - 50 - totalH) / 2);
      for (let ri = 0; ri < colNodes.length; ri++) {
        const node = colNodes[ri];
        rects.set(node.id, {
          x: startX + ci * (cardW + gapX),
          y: startY + ri * (cardH + gapY),
          w: cardW,
          h: cardH,
          node,
        });
      }
    }

    // Draw edges
    const edgeColors: Record<string, string> = {
      structure: '#3b82f6',
      sequences: '#8b5cf6',
      folds: '#10b981',
      metrics: '#f59e0b',
    };
    for (const e of graph.edges) {
      const src = rects.get(e.fromNode);
      const dst = rects.get(e.toNode);
      if (!src || !dst) continue;
      const x0 = src.x + src.w;
      const y0 = src.y + src.h / 2;
      const x1 = dst.x;
      const y1 = dst.y + dst.h / 2;
      const dx = Math.max(32, Math.abs(x1 - x0) * 0.52);
      const col = edgeColors[e.fromPort] ?? '#64748b';
      ctx.strokeStyle = col;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(x0 + dx, y0, x1 - dx, y1, x1 - 4, y1);
      ctx.stroke();

      // Arrowhead
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(x1 - 1, y1);
      ctx.lineTo(x1 - 10, y1 - 5);
      ctx.lineTo(x1 - 10, y1 + 5);
      ctx.closePath();
      ctx.fill();
    }

    // Draw cards
    for (const r of rects.values()) {
      const manifest = ADAPTER_REGISTRY[r.node.adapterId];
      const cat = STAGE_COLORS[manifest?.stage ?? 'score'] ?? STAGE_COLORS.score;
      const label = r.node.label || manifest?.shortName || r.node.adapterId;
      const paramChip = formatNodeParamSummary(r.node.adapterId, r.node.params);
      const resultBadge = summarizeNodeResultBadge(r.node, outputsByNode[r.node.id]);

      ctx.save();
      ctx.shadowColor = isDark ? 'rgba(0,0,0,0.35)' : 'rgba(15, 23, 42, 0.08)';
      ctx.shadowBlur = 12;
      ctx.shadowOffsetY = 4;
      roundRectPath(ctx, r.x, r.y, r.w, r.h, 13);
      ctx.fillStyle = isDark ? '#1e293b' : '#ffffff';
      ctx.fill();
      ctx.restore();

      ctx.save();
      roundRectPath(ctx, r.x, r.y, r.w, r.h, 13);
      ctx.clip();
      ctx.fillStyle = cat.accent;
      ctx.fillRect(r.x, r.y, r.w, 5);
      ctx.restore();

      roundRectPath(ctx, r.x, r.y, r.w, r.h, 13);
      ctx.strokeStyle = resultBadge ? '#10b981' : isDark ? '#334155' : '#cbd5e1';
      ctx.lineWidth = resultBadge ? 1.8 : 1.3;
      ctx.stroke();

      ctx.fillStyle = cat.accent;
      ctx.font = '700 10.5px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
      ctx.fillText(cat.label, r.x + 14, r.y + 23);

      ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';
      ctx.font = '700 15.5px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
      drawEllipsis(ctx, label, r.x + 14, r.y + 45, r.w - 28);

      // Param chip
      const chipY = r.y + 56;
      roundRectPath(ctx, r.x + 12, chipY, r.w - 24, 23, 6);
      ctx.fillStyle = isDark ? '#0f172a' : '#f1f5f9';
      ctx.fill();
      ctx.fillStyle = isDark ? '#cbd5e1' : '#475569';
      ctx.font = '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
      drawEllipsis(ctx, paramChip, r.x + 19, chipY + 15.5, r.w - 38);

      if (resultBadge) {
        ctx.fillStyle = isDark ? '#34d399' : '#047857';
        ctx.font = '600 11px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
        drawEllipsis(ctx, `✓ ${resultBadge}`, r.x + 14, r.y + r.h - 11, r.w - 28);
      }
    }
  }
  ctx.restore();

  roundRectPath(ctx, rx, ry, rw, rh, 18);
  ctx.strokeStyle = isDark ? '#1f2937' : '#e2e8f0';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

/**
 * Compose a 1200x630 social-media PNG card in the browser combining the live Mol* 3D
 * final structure render and/or the workflow DAG canvas with run results.
 */
export async function renderClientSocialCardPng(opts: SocialCardOptions): Promise<{
  dataUrl: string;
  blob: Blob | null;
}> {
  const isDark = opts.theme === 'dark';
  const canvas = document.createElement('canvas');
  canvas.width = CARD_W;
  canvas.height = CARD_H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas2D context unavailable');

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, 0, CARD_H);
  if (isDark) {
    grad.addColorStop(0, '#0b1120');
    grad.addColorStop(1, '#111827');
  } else {
    grad.addColorStop(0, '#f8fafc');
    grad.addColorStop(1, '#eef2f9');
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, CARD_W, CARD_H);

  // Header bar
  ctx.fillStyle = isDark ? '#0f172a' : '#ffffff';
  ctx.fillRect(0, 0, CARD_W, 78);
  ctx.fillStyle = isDark ? '#1e293b' : '#e2e8f0';
  ctx.fillRect(0, 77, CARD_W, 1);

  drawBrandMark(ctx, 30, 18, 42);
  ctx.font = '700 26px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
  ctx.fillStyle = isDark ? '#60a5fa' : '#2563eb';
  ctx.fillText('denovo', 84, 48);
  const dw = ctx.measureText('denovo').width;
  ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';
  ctx.fillText('.run', 84 + dw, 48);

  // Header pill
  const pillText = 'Browser-Native WebGPU · 100% Local';
  ctx.font = '600 13px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
  const pw = ctx.measureText(pillText).width + 28;
  roundRectPath(ctx, CARD_W - 30 - pw, 23, pw, 32, 16);
  ctx.fillStyle = isDark ? 'rgba(37, 99, 235, 0.18)' : '#eff6ff';
  ctx.fill();
  ctx.strokeStyle = isDark ? 'rgba(96, 165, 250, 0.35)' : '#bfdbfe';
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.fillStyle = isDark ? '#93c5fd' : '#1d4ed8';
  ctx.fillText(pillText, CARD_W - 30 - pw + 14, 44);

  const img = opts.structurePngDataUrl ? await loadImage(opts.structurePngDataUrl) : null;
  const effectiveMode: SocialCardMode = !img && opts.mode !== 'canvas' ? 'canvas' : opts.mode;

  const bodyY = 96;
  const bodyH = 474;

  const drawStructurePanel = (sx: number, sy: number, sw: number, sh: number) => {
    ctx.save();
    ctx.shadowColor = 'rgba(15, 23, 42, 0.14)';
    ctx.shadowBlur = 18;
    ctx.shadowOffsetY = 5;
    roundRectPath(ctx, sx, sy, sw, sh, 18);
    ctx.fillStyle = isDark ? '#0d0e11' : '#f8fafc';
    ctx.fill();
    ctx.restore();

    ctx.save();
    roundRectPath(ctx, sx, sy, sw, sh, 18);
    ctx.clip();
    if (img) {
      // Aspect-preserving cover/contain fit with clean padding
      const scale = Math.min(sw / img.width, sh / img.height);
      const dwImg = img.width * scale;
      const dhImg = img.height * scale;
      ctx.drawImage(img, sx + (sw - dwImg) / 2, sy + (sh - dhImg) / 2, dwImg, dhImg);
    }

    // Top-left label pill
    if (opts.structureLabel) {
      ctx.font = '600 13px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
      const lblW = Math.min(sw - 32, ctx.measureText(opts.structureLabel).width + 34);
      roundRectPath(ctx, sx + 16, sy + 16, lblW, 30, 9);
      ctx.fillStyle = isDark ? 'rgba(15, 23, 42, 0.86)' : 'rgba(255, 255, 255, 0.92)';
      ctx.fill();
      ctx.strokeStyle = isDark ? '#334155' : '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx + 29, sy + 31, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981';
      ctx.fill();
      ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';
      drawEllipsis(ctx, opts.structureLabel, sx + 40, sy + 35.5, lblW - 30);
    }

    if (opts.structureMeta) {
      ctx.font = '500 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      const metaW = Math.min(sw - 32, ctx.measureText(opts.structureMeta).width + 24);
      roundRectPath(ctx, sx + 16, sy + sh - 40, metaW, 26, 7);
      ctx.fillStyle = isDark ? 'rgba(15, 23, 42, 0.84)' : 'rgba(255, 255, 255, 0.90)';
      ctx.fill();
      ctx.fillStyle = isDark ? '#cbd5e1' : '#334155';
      drawEllipsis(ctx, opts.structureMeta, sx + 28, sy + sh - 23, metaW - 20);
    }
    ctx.restore();

    roundRectPath(ctx, sx, sy, sw, sh, 18);
    ctx.strokeStyle = isDark ? '#1e293b' : '#cbd5e1';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  if (effectiveMode === 'both' && img) {
    const isSingle = opts.graph.nodes.length <= 1;
    const dagW = isSingle ? 450 : 660;
    const structX = 30 + dagW + 20;
    const structW = CARD_W - structX - 30;
    drawDagPanel(ctx, opts.graph, opts.outputsByNode, 30, bodyY, dagW, bodyH, isDark);
    drawStructurePanel(structX, bodyY, structW, bodyH);
  } else if (effectiveMode === 'structure' && img) {
    drawStructurePanel(30, bodyY, CARD_W - 60, bodyH);
  } else {
    drawDagPanel(ctx, opts.graph, opts.outputsByNode, 30, bodyY, CARD_W - 60, bodyH, isDark);
  }

  // Footer
  const chainTitle =
    opts.graph.nodes.length > 0
      ? opts.graph.nodes
          .slice(0, 4)
          .map((n) => n.label || ADAPTER_REGISTRY[n.adapterId]?.shortName || n.adapterId)
          .join(' → ')
      : opts.graph.name || 'Custom Workflow';
  ctx.font = '600 14px -apple-system, BlinkMacSystemFont, Inter, "Segoe UI", sans-serif';
  ctx.fillStyle = isDark ? '#e2e8f0' : '#0f172a';
  drawEllipsis(ctx, chainTitle, 32, CARD_H - 22, CARD_W - 260);

  const rightText = `${opts.graph.nodes.length} block${opts.graph.nodes.length === 1 ? '' : 's'} · denovo.run`;
  ctx.fillStyle = isDark ? '#60a5fa' : '#2563eb';
  const rtw = ctx.measureText(rightText).width;
  ctx.fillText(rightText, CARD_W - 32 - rtw, CARD_H - 22);

  const dataUrl = canvas.toDataURL('image/png');
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), 'image/png'));
  return { dataUrl, blob };
}
