import { OG_FONT_ATLAS, type OgFontKey } from './font-atlas';

export type Rgba = readonly [number, number, number, number];

export interface ProteinChainTrace {
  id: string;
  points: Array<readonly [number, number, number]>;
  plddt?: number[];
}

export interface ProteinLigandTrace {
  name: string;
  chain: string;
  atoms: Array<readonly [number, number, number]>;
}

export interface EdgeStructurePreview {
  title: string;
  subtitle: string;
  chains: ProteinChainTrace[];
  ligands: ProteinLigandTrace[];
  residueCount: number;
  isPlddt: boolean;
}

const decodedFontCache = new Map<OgFontKey, Uint8Array>();

function getFontBytes(key: OgFontKey): Uint8Array {
  let bytes = decodedFontCache.get(key);
  if (bytes) return bytes;
  const b64 = OG_FONT_ATLAS[key].b64;
  const bin = atob(b64);
  bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  decodedFontCache.set(key, bytes);
  return bytes;
}

export function hexToRgba(hex: string, a = 255): Rgba {
  const clean = hex.replace('#', '');
  const n = parseInt(clean.length === 3 ? clean.replace(/(.)/g, '$1$1') : clean, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, a];
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Pure-JS 2D/3D RGBA framebuffer with smooth antialiased shapes, typography,
 * Catmull-Rom 3D protein ribbon rendering, and built-in CompressionStream PNG encoding.
 */
export class OgCanvas {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;

  constructor(width: number, height: number, bgHex = '#f8fafc') {
    this.width = width;
    this.height = height;
    this.data = new Uint8Array(width * height * 4);
    const [r, g, b, a] = hexToRgba(bgHex, 255);
    for (let i = 0; i < this.data.length; i += 4) {
      this.data[i] = r;
      this.data[i + 1] = g;
      this.data[i + 2] = b;
      this.data[i + 3] = a;
    }
  }

  blendPixel(x: number, y: number, r: number, g: number, b: number, alpha01: number): void {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height || alpha01 <= 0.002) return;
    const idx = (y * this.width + x) * 4;
    if (alpha01 >= 0.996) {
      this.data[idx] = r;
      this.data[idx + 1] = g;
      this.data[idx + 2] = b;
      this.data[idx + 3] = 255;
      return;
    }
    const inv = 1 - alpha01;
    this.data[idx] = Math.round(r * alpha01 + this.data[idx] * inv);
    this.data[idx + 1] = Math.round(g * alpha01 + this.data[idx + 1] * inv);
    this.data[idx + 2] = Math.round(b * alpha01 + this.data[idx + 2] * inv);
    this.data[idx + 3] = 255;
  }

  fillLinearGradientRect(
    x: number,
    y: number,
    w: number,
    h: number,
    topHex: string,
    bottomHex: string,
  ): void {
    const c0 = hexToRgba(topHex);
    const c1 = hexToRgba(bottomHex);
    const x0 = Math.max(0, Math.floor(x));
    const y0 = Math.max(0, Math.floor(y));
    const x1 = Math.min(this.width, Math.ceil(x + w));
    const y1 = Math.min(this.height, Math.ceil(y + h));
    for (let py = y0; py < y1; py++) {
      const t = h > 1 ? clamp((py - y) / (h - 1), 0, 1) : 0;
      const r = Math.round(lerp(c0[0], c1[0], t));
      const g = Math.round(lerp(c0[1], c1[1], t));
      const b = Math.round(lerp(c0[2], c1[2], t));
      for (let px = x0; px < x1; px++) {
        const idx = (py * this.width + px) * 4;
        this.data[idx] = r;
        this.data[idx + 1] = g;
        this.data[idx + 2] = b;
        this.data[idx + 3] = 255;
      }
    }
  }

  drawDotGrid(
    x: number,
    y: number,
    w: number,
    h: number,
    spacing = 22,
    dotHex = '#cbd5e1',
    alpha = 0.65,
  ): void {
    const [r, g, b] = hexToRgba(dotHex);
    for (let gy = Math.floor(y) + spacing / 2; gy < y + h; gy += spacing) {
      for (let gx = Math.floor(x) + spacing / 2; gx < x + w; gx += spacing) {
        this.fillCircle(gx, gy, 1.15, [r, g, b, Math.round(alpha * 255)]);
      }
    }
  }

  private roundedRectSdf(
    px: number,
    py: number,
    cx: number,
    cy: number,
    hw: number,
    hh: number,
    rad: number,
  ): number {
    const qx = Math.abs(px - cx) - (hw - rad);
    const qy = Math.abs(py - cy) - (hh - rad);
    const ox = Math.max(qx, 0);
    const oy = Math.max(qy, 0);
    return Math.hypot(ox, oy) + Math.min(Math.max(qx, qy), 0) - rad;
  }

  drawCardShadow(x: number, y: number, w: number, h: number, radius: number, spread = 12): void {
    const rad = Math.min(radius, w / 2, h / 2);
    const cx = x + w / 2;
    const cy = y + h / 2 + 4;
    const hw = w / 2;
    const hh = h / 2;
    const x0 = Math.max(0, Math.floor(x - spread));
    const y0 = Math.max(0, Math.floor(y - spread + 4));
    const x1 = Math.min(this.width - 1, Math.ceil(x + w + spread));
    const y1 = Math.min(this.height - 1, Math.ceil(y + h + spread + 4));
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d = this.roundedRectSdf(px + 0.5, py + 0.5, cx, cy, hw, hh, rad);
        if (d > spread) continue;
        if (d <= 0) {
          this.blendPixel(px, py, 15, 23, 42, 0.08);
        } else {
          const t = 1 - d / spread;
          this.blendPixel(px, py, 15, 23, 42, 0.08 * t * t);
        }
      }
    }
  }

  fillRoundedRect(
    x: number,
    y: number,
    w: number,
    h: number,
    radius: number,
    color: Rgba,
  ): void {
    if (w <= 0 || h <= 0) return;
    const rad = Math.min(radius, w / 2, h / 2);
    const cx = x + w / 2;
    const cy = y + h / 2;
    const hw = w / 2;
    const hh = h / 2;
    const x0 = Math.max(0, Math.floor(x - 1));
    const y0 = Math.max(0, Math.floor(y - 1));
    const x1 = Math.min(this.width - 1, Math.ceil(x + w + 1));
    const y1 = Math.min(this.height - 1, Math.ceil(y + h + 1));
    const baseA = color[3] / 255;
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d = this.roundedRectSdf(px + 0.5, py + 0.5, cx, cy, hw, hh, rad);
        if (d >= 0.75) continue;
        const cov = d <= -0.75 ? 1 : clamp(0.5 - d / 1.5, 0, 1);
        this.blendPixel(px, py, color[0], color[1], color[2], cov * baseA);
      }
    }
  }

  /** Fill only the top rounded strip of a card (accent bar). */
  fillTopRoundedBar(
    x: number,
    y: number,
    w: number,
    barHeight: number,
    cardHeight: number,
    radius: number,
    color: Rgba,
  ): void {
    if (w <= 0 || barHeight <= 0) return;
    const rad = Math.min(radius, w / 2, cardHeight / 2);
    const cx = x + w / 2;
    const cy = y + cardHeight / 2;
    const hw = w / 2;
    const hh = cardHeight / 2;
    const x0 = Math.max(0, Math.floor(x - 1));
    const y0 = Math.max(0, Math.floor(y - 1));
    const x1 = Math.min(this.width - 1, Math.ceil(x + w + 1));
    const y1 = Math.min(this.height - 1, Math.ceil(y + barHeight));
    const baseA = color[3] / 255;
    for (let py = y0; py < y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d = this.roundedRectSdf(px + 0.5, py + 0.5, cx, cy, hw, hh, rad);
        if (d >= 0.75) continue;
        const cov = d <= -0.75 ? 1 : clamp(0.5 - d / 1.5, 0, 1);
        this.blendPixel(px, py, color[0], color[1], color[2], cov * baseA);
      }
    }
  }

  strokeRoundedRect(
    x: number,
    y: number,
    w: number,
    h: number,
    radius: number,
    strokeWidth: number,
    color: Rgba,
  ): void {
    const rad = Math.min(radius, w / 2, h / 2);
    const cx = x + w / 2;
    const cy = y + h / 2;
    const hw = w / 2;
    const hh = h / 2;
    const halfS = strokeWidth / 2;
    const pad = Math.ceil(halfS + 2);
    const x0 = Math.max(0, Math.floor(x - pad));
    const y0 = Math.max(0, Math.floor(y - pad));
    const x1 = Math.min(this.width - 1, Math.ceil(x + w + pad));
    const y1 = Math.min(this.height - 1, Math.ceil(y + h + pad));
    const baseA = color[3] / 255;
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d = Math.abs(this.roundedRectSdf(px + 0.5, py + 0.5, cx, cy, hw, hh, rad)) - halfS;
        if (d >= 0.75) continue;
        const cov = d <= -0.75 ? 1 : clamp(0.5 - d / 1.5, 0, 1);
        this.blendPixel(px, py, color[0], color[1], color[2], cov * baseA);
      }
    }
  }

  fillCircle(cx: number, cy: number, r: number, color: Rgba): void {
    const x0 = Math.max(0, Math.floor(cx - r - 1));
    const y0 = Math.max(0, Math.floor(cy - r - 1));
    const x1 = Math.min(this.width - 1, Math.ceil(cx + r + 1));
    const y1 = Math.min(this.height - 1, Math.ceil(cy + r + 1));
    const baseA = color[3] / 255;
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const d = Math.hypot(px + 0.5 - cx, py + 0.5 - cy) - r;
        if (d >= 0.75) continue;
        const cov = d <= -0.75 ? 1 : clamp(0.5 - d / 1.5, 0, 1);
        this.blendPixel(px, py, color[0], color[1], color[2], cov * baseA);
      }
    }
  }

  drawSegment(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    radius: number,
    color: Rgba,
  ): void {
    const minX = Math.max(0, Math.floor(Math.min(x0, x1) - radius - 1));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1) - radius - 1));
    const maxX = Math.min(this.width - 1, Math.ceil(Math.max(x0, x1) + radius + 1));
    const maxY = Math.min(this.height - 1, Math.ceil(Math.max(y0, y1) + radius + 1));
    const dx = x1 - x0;
    const dy = y1 - y0;
    const lenSq = dx * dx + dy * dy;
    const baseA = color[3] / 255;
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const rx = px + 0.5 - x0;
        const ry = py + 0.5 - y0;
        const t = lenSq > 1e-6 ? clamp((rx * dx + ry * dy) / lenSq, 0, 1) : 0;
        const d = Math.hypot(rx - dx * t, ry - dy * t) - radius;
        if (d >= 0.75) continue;
        const cov = d <= -0.75 ? 1 : clamp(0.5 - d / 1.5, 0, 1);
        this.blendPixel(px, py, color[0], color[1], color[2], cov * baseA);
      }
    }
  }

  /** Draw a shaded cylindrical capsule (used for 3D protein ribbon rendering). */
  drawShadedCapsule(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    radius: number,
    baseColor: readonly [number, number, number],
    outlineColor: readonly [number, number, number],
    depthShade: number,
  ): void {
    const outerR = radius + 1.1;
    const minX = Math.max(0, Math.floor(Math.min(x0, x1) - outerR - 1));
    const minY = Math.max(0, Math.floor(Math.min(y0, y1) - outerR - 1));
    const maxX = Math.min(this.width - 1, Math.ceil(Math.max(x0, x1) + outerR + 1));
    const maxY = Math.min(this.height - 1, Math.ceil(Math.max(y0, y1) + outerR + 1));
    const dx = x1 - x0;
    const dy = y1 - y0;
    const lenSq = dx * dx + dy * dy;
    const len = Math.sqrt(lenSq) || 1;
    // Perpendicular unit vector pointing slightly up-left for cylindrical lighting
    const nx = -dy / len;
    const ny = dx / len;
    const lDotScale = -0.55 * nx - 0.83 * ny;

    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const rx = px + 0.5 - x0;
        const ry = py + 0.5 - y0;
        const t = lenSq > 1e-6 ? clamp((rx * dx + ry * dy) / lenSq, 0, 1) : 0;
        const ox = rx - dx * t;
        const oy = ry - dy * t;
        const dist = Math.hypot(ox, oy);
        if (dist >= outerR + 0.7) continue;

        const outerCov = clamp(0.5 - (dist - outerR) / 1.4, 0, 1);
        if (dist > radius) {
          this.blendPixel(px, py, outlineColor[0], outlineColor[1], outlineColor[2], outerCov * 0.85);
          continue;
        }
        const radial = dist / Math.max(0.5, radius);
        const zCyl = Math.sqrt(Math.max(0, 1 - radial * radial));
        const signedSide = (ox * nx + oy * ny) / Math.max(0.5, radius);
        const light = clamp(0.44 + 0.44 * zCyl + 0.22 * signedSide * (lDotScale >= 0 ? 1 : -1), 0.25, 1.15);
        const spec = Math.pow(clamp(zCyl * 0.92 + Math.max(0, -oy / Math.max(0.5, radius)) * 0.28, 0, 1), 10) * 65;
        const s = light * depthShade;
        const r = clamp(Math.round(baseColor[0] * s + spec), 0, 255);
        const g = clamp(Math.round(baseColor[1] * s + spec), 0, 255);
        const b = clamp(Math.round(baseColor[2] * s + spec), 0, 255);
        this.blendPixel(px, py, r, g, b, outerCov);
      }
    }
  }

  /** Draw a shaded 3D sphere (for ligands, ions, and terminal atoms). */
  drawShadedSphere(
    cx: number,
    cy: number,
    radius: number,
    baseColor: readonly [number, number, number],
    outlineColor: readonly [number, number, number],
    depthShade: number,
  ): void {
    const outerR = radius + 1.1;
    const x0 = Math.max(0, Math.floor(cx - outerR - 1));
    const y0 = Math.max(0, Math.floor(cy - outerR - 1));
    const x1 = Math.min(this.width - 1, Math.ceil(cx + outerR + 1));
    const y1 = Math.min(this.height - 1, Math.ceil(cy + outerR + 1));
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const ox = px + 0.5 - cx;
        const oy = py + 0.5 - cy;
        const dist = Math.hypot(ox, oy);
        if (dist >= outerR + 0.7) continue;
        const cov = clamp(0.5 - (dist - outerR) / 1.4, 0, 1);
        if (dist > radius) {
          this.blendPixel(px, py, outlineColor[0], outlineColor[1], outlineColor[2], cov * 0.9);
          continue;
        }
        const nx = ox / Math.max(0.5, radius);
        const ny = oy / Math.max(0.5, radius);
        const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
        const ndl = clamp(-0.42 * nx - 0.58 * ny + 0.7 * nz, 0, 1);
        const light = (0.36 + 0.74 * ndl) * depthShade;
        const spec = Math.pow(clamp(-0.3 * nx - 0.4 * ny + 0.86 * nz, 0, 1), 16) * 115;
        const r = clamp(Math.round(baseColor[0] * light + spec), 0, 255);
        const g = clamp(Math.round(baseColor[1] * light + spec), 0, 255);
        const b = clamp(Math.round(baseColor[2] * light + spec), 0, 255);
        this.blendPixel(px, py, r, g, b, cov);
      }
    }
  }

  strokeCubicBezier(
    x0: number,
    y0: number,
    cx1: number,
    cy1: number,
    cx2: number,
    cy2: number,
    x1: number,
    y1: number,
    radius: number,
    color: Rgba,
    withArrow = false,
  ): void {
    const steps = 28;
    let px = x0;
    let py = y0;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const mt = 1 - t;
      const nx =
        mt * mt * mt * x0 +
        3 * mt * mt * t * cx1 +
        3 * mt * t * t * cx2 +
        t * t * t * x1;
      const ny =
        mt * mt * mt * y0 +
        3 * mt * mt * t * cy1 +
        3 * mt * t * t * cy2 +
        t * t * t * y1;
      this.drawSegment(px, py, nx, ny, radius, color);
      px = nx;
      py = ny;
    }
    if (withArrow) {
      const ang = Math.atan2(y1 - cy2, x1 - cx2);
      const headLen = 9;
      const a1 = ang - 2.62;
      const a2 = ang + 2.62;
      this.drawSegment(x1, y1, x1 + Math.cos(a1) * headLen, y1 + Math.sin(a1) * headLen, radius * 0.95, color);
      this.drawSegment(x1, y1, x1 + Math.cos(a2) * headLen, y1 + Math.sin(a2) * headLen, radius * 0.95, color);
    }
  }

  measureText(text: string, fontKey: OgFontKey): number {
    const font = OG_FONT_ATLAS[fontKey];
    let w = 0;
    for (const ch of text) {
      const g = font.glyphs[ch] ?? font.glyphs['?'] ?? font.glyphs[' '];
      w += g ? g[0] : Math.round(font.size * 0.55);
    }
    return w;
  }

  drawText(text: string, x: number, baselineY: number, fontKey: OgFontKey, color: Rgba): number {
    const font = OG_FONT_ATLAS[fontKey];
    const bytes = getFontBytes(fontKey);
    const baseA = color[3] / 255;
    let cursorX = Math.round(x);
    const by = Math.round(baselineY);

    for (const ch of text) {
      const g = font.glyphs[ch] ?? font.glyphs['?'] ?? font.glyphs[' '];
      if (!g) {
        cursorX += Math.round(font.size * 0.55);
        continue;
      }
      const [adv, ox, oy, w, h, byteOffset] = g;
      if (w > 0 && h > 0) {
        let p = 0;
        for (let gy = 0; gy < h; gy++) {
          const py = by + oy + gy;
          for (let gx = 0; gx < w; gx++) {
            const byte = bytes[byteOffset + (p >> 1)];
            const q = (p & 1) === 0 ? byte >> 4 : byte & 0x0f;
            if (q > 0) {
              this.blendPixel(cursorX + ox + gx, py, color[0], color[1], color[2], (q / 15) * baseA);
            }
            p++;
          }
        }
      }
      cursorX += adv;
    }
    return cursorX - Math.round(x);
  }

  drawEllipsisText(
    text: string,
    x: number,
    baselineY: number,
    maxWidth: number,
    fontKey: OgFontKey,
    color: Rgba,
  ): number {
    if (this.measureText(text, fontKey) <= maxWidth) {
      return this.drawText(text, x, baselineY, fontKey, color);
    }
    const ell = '…';
    const ellW = this.measureText(ell, fontKey);
    let out = '';
    for (const ch of text) {
      if (this.measureText(out + ch, fontKey) + ellW > maxWidth) break;
      out += ch;
    }
    return this.drawText(out.trimEnd() + ell, x, baselineY, fontKey, color);
  }

  /** Render the denovo.run 3-dot ribbon logo mark cleanly inside a square box. */
  drawBrandLogo(x: number, y: number, size: number): void {
    const s = size / 32;
    this.fillRoundedRect(x, y, size, size, 7 * s, hexToRgba('#0f172a'));
    this.strokeCubicBezier(
      x + 8 * s,
      y + 22 * s,
      x + 11 * s,
      y + 10 * s,
      x + 15 * s,
      y + 10 * s,
      x + 16 * s,
      y + 16 * s,
      1.5 * s,
      hexToRgba('#60a5fa'),
      false,
    );
    this.strokeCubicBezier(
      x + 16 * s,
      y + 16 * s,
      x + 17 * s,
      y + 22 * s,
      x + 21 * s,
      y + 22 * s,
      x + 24 * s,
      y + 10 * s,
      1.5 * s,
      hexToRgba('#60a5fa'),
      false,
    );
    this.fillCircle(x + 8 * s, y + 22 * s, 2.7 * s, hexToRgba('#38bdf8'));
    this.fillCircle(x + 16 * s, y + 16 * s, 2.7 * s, hexToRgba('#a78bfa'));
    this.fillCircle(x + 24 * s, y + 10 * s, 2.7 * s, hexToRgba('#34d399'));
  }

  /**
   * Render a 3D protein structure (Catmull-Rom Cα ribbon + ball-and-stick ligands)
   * using PCA optimal framing so the broadest face fills the viewport.
   */
  renderProtein3D(
    structure: EdgeStructurePreview,
    boxX: number,
    boxY: number,
    boxW: number,
    boxH: number,
  ): void {
    const allPts: Array<readonly [number, number, number]> = [];
    for (const ch of structure.chains) {
      for (const pt of ch.points) allPts.push(pt);
    }
    for (const lig of structure.ligands) {
      for (const pt of lig.atoms) allPts.push(pt);
    }
    if (allPts.length === 0) return;

    // 1. Centroid
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const p of allPts) {
      cx += p[0];
      cy += p[1];
      cz += p[2];
    }
    cx /= allPts.length;
    cy /= allPts.length;
    cz /= allPts.length;

    // 2. 3x3 symmetric covariance + Jacobi eigendecomposition for optimal framing
    let c00 = 0,
      c01 = 0,
      c02 = 0,
      c11 = 0,
      c12 = 0,
      c22 = 0;
    for (const p of allPts) {
      const dx = p[0] - cx;
      const dy = p[1] - cy;
      const dz = p[2] - cz;
      c00 += dx * dx;
      c01 += dx * dy;
      c02 += dx * dz;
      c11 += dy * dy;
      c12 += dy * dz;
      c22 += dz * dz;
    }
    const basis = jacobiPrincipalAxes(c00, c01, c02, c11, c12, c22);
    // Align longest principal axis with viewport's longer dimension (portrait vs landscape)
    const orientPortrait = boxH > boxW * 1.05;
    const project3 = (p: readonly [number, number, number]): [number, number, number] => {
      const dx = p[0] - cx;
      const dy = p[1] - cy;
      const dz = p[2] - cz;
      const p0 = dx * basis[0][0] + dy * basis[0][1] + dz * basis[0][2];
      const p1 = dx * basis[1][0] + dy * basis[1][1] + dz * basis[1][2];
      const p2 = dx * basis[2][0] + dy * basis[2][1] + dz * basis[2][2];
      const x0 = orientPortrait ? p1 : p0;
      const y0 = orientPortrait ? p0 : p1;
      const z0 = p2;
      const ct = 0.978;
      const st = 0.208;
      const y1 = y0 * ct - z0 * st;
      const z1 = y0 * st + z0 * ct;
      return [x0, y1, z1];
    };

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity,
      minZ = Infinity,
      maxZ = -Infinity;
    const updateBounds = (q: readonly [number, number, number]) => {
      if (q[0] < minX) minX = q[0];
      if (q[0] > maxX) maxX = q[0];
      if (q[1] < minY) minY = q[1];
      if (q[1] > maxY) maxY = q[1];
      if (q[2] < minZ) minZ = q[2];
      if (q[2] > maxZ) maxZ = q[2];
    };

    const projChains = structure.chains.map((ch) => {
      const pts = ch.points.map((p) => {
        const q = project3(p);
        updateBounds(q);
        return q;
      });
      return { id: ch.id, pts, plddt: ch.plddt };
    });
    const projLigands = structure.ligands.map((lig) => {
      const pts = lig.atoms.map((p) => {
        const q = project3(p);
        updateBounds(q);
        return q;
      });
      return { name: lig.name, pts };
    });

    const spanX = Math.max(8, maxX - minX);
    const spanY = Math.max(8, maxY - minY);
    const spanZ = Math.max(8, maxZ - minZ);
    const pad = 34;
    const availW = Math.max(40, boxW - pad * 2);
    const availH = Math.max(40, boxH - pad * 2);
    const scale = Math.min(availW / spanX, availH / spanY);
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const midZ = (minZ + maxZ) / 2;
    const centerX = boxX + boxW / 2;
    const centerY = boxY + boxH / 2;

    const toScreen = (q: readonly [number, number, number]): [number, number, number] => {
      const zNorm = (q[2] - midZ) / Math.max(12, spanZ);
      const persp = 1 + zNorm * 0.14;
      const sx = centerX + (q[0] - midX) * scale * persp;
      const sy = centerY - (q[1] - midY) * scale * persp;
      return [sx, sy, zNorm];
    };

    type DrawItem =
      | {
          kind: 'capsule';
          z: number;
          x0: number;
          y0: number;
          x1: number;
          y1: number;
          r: number;
          color: [number, number, number];
          outline: [number, number, number];
        }
      | {
          kind: 'sphere';
          z: number;
          x: number;
          y: number;
          r: number;
          color: [number, number, number];
          outline: [number, number, number];
        };

    const drawList: DrawItem[] = [];
    const totalRes = Math.max(1, structure.residueCount);
    const ribbonRadius = clamp(scale * 0.72, 2.3, 6.2);
    const multiChain = projChains.length > 1;
    const chainPalette: Array<[number, number, number]> = [
      [14, 165, 233], // sky
      [139, 92, 246], // violet
      [16, 185, 129], // emerald
      [245, 158, 11], // amber
      [236, 72, 153], // pink
      [59, 130, 246], // blue
    ];

    const subSteps = totalRes > 400 ? 2 : 3;

    for (let cIdx = 0; cIdx < projChains.length; cIdx++) {
      const ch = projChains[cIdx];
      const n = ch.pts.length;
      if (n < 2) continue;

      for (let i = 0; i < n - 1; i++) {
        const p1 = ch.pts[i];
        const p2 = ch.pts[i + 1];
        // Break ribbon cleanly across chain gaps (> 6.5 Å between Cα)
        const caDist = Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
        if (caDist > 6.8) continue;

        const p0 = i > 0 && dist3(ch.pts[i - 1], p1) <= 6.8 ? ch.pts[i - 1] : p1;
        const p3 = i + 2 < n && dist3(p2, ch.pts[i + 2]) <= 6.8 ? ch.pts[i + 2] : p2;

        const baseRgb: [number, number, number] = structure.isPlddt && ch.plddt
          ? plddtToRgb(ch.plddt[i] ?? 80)
          : multiChain
            ? chainPalette[cIdx % chainPalette.length]
            : spectrumRgb(i / Math.max(1, n - 1));
        const outlineRgb: [number, number, number] = [
          Math.round(baseRgb[0] * 0.34),
          Math.round(baseRgb[1] * 0.34),
          Math.round(baseRgb[2] * 0.34),
        ];

        let prev = toScreen(p1);
        for (let s = 1; s <= subSteps; s++) {
          const t = s / subSteps;
          const pt = catmullRom3(p0, p1, p2, p3, t);
          const cur = toScreen(pt);
          const zMid = (prev[2] + cur[2]) * 0.5;
          drawList.push({
            kind: 'capsule',
            z: zMid,
            x0: prev[0],
            y0: prev[1],
            x1: cur[0],
            y1: cur[1],
            r: ribbonRadius * (0.94 + zMid * 0.14),
            color: baseRgb,
            outline: outlineRgb,
          });
          prev = cur;
        }
      }
    }

    // Add ball-and-stick ligand atoms
    const ligRadius = clamp(ribbonRadius * 1.15, 3.6, 7.6);
    for (const lig of projLigands) {
      const screenPts = lig.pts.map(toScreen);
      for (let i = 0; i < screenPts.length; i++) {
        for (let j = i + 1; j < screenPts.length; j++) {
          if (dist3(lig.pts[i], lig.pts[j]) <= 1.95) {
            const zMid = (screenPts[i][2] + screenPts[j][2]) * 0.5;
            drawList.push({
              kind: 'capsule',
              z: zMid - 0.005,
              x0: screenPts[i][0],
              y0: screenPts[i][1],
              x1: screenPts[j][0],
              y1: screenPts[j][1],
              r: ligRadius * 0.42,
              color: [249, 115, 22],
              outline: [124, 45, 18],
            });
          }
        }
        drawList.push({
          kind: 'sphere',
          z: screenPts[i][2] + 0.01,
          x: screenPts[i][0],
          y: screenPts[i][1],
          r: ligRadius,
          color: [249, 115, 22],
          outline: [124, 45, 18],
        });
      }
    }

    drawList.sort((a, b) => a.z - b.z);
    for (const item of drawList) {
      const depthShade = clamp(0.76 + (item.z + 0.5) * 0.34, 0.65, 1.12);
      if (item.kind === 'capsule') {
        this.drawShadedCapsule(item.x0, item.y0, item.x1, item.y1, item.r, item.color, item.outline, depthShade);
      } else {
        this.drawShadedSphere(item.x, item.y, item.r, item.color, item.outline, depthShade);
      }
    }
  }

  /** Encode framebuffer to a standard RGBA PNG using native CompressionStream('deflate'). */
  async encodePng(): Promise<Uint8Array> {
    const stride = this.width * 4;
    const raw = new Uint8Array((stride + 1) * this.height);
    for (let y = 0; y < this.height; y++) {
      const dstRow = y * (stride + 1);
      const srcRow = y * stride;
      // Sub filter (type 1) compresses flat UI backgrounds & cards much smaller than filter 0
      raw[dstRow] = 1;
      for (let x = 0; x < stride; x++) {
        const left = x >= 4 ? this.data[srcRow + x - 4] : 0;
        raw[dstRow + 1 + x] = (this.data[srcRow + x] - left) & 0xff;
      }
    }

    const compressed = await deflateZlib(raw);
    const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

    const ihdr = new Uint8Array(13);
    const dv = new DataView(ihdr.buffer);
    dv.setUint32(0, this.width, false);
    dv.setUint32(4, this.height, false);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // RGBA
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    const cIhdr = pngChunk('IHDR', ihdr);
    const cIdat = pngChunk('IDAT', compressed);
    const cIend = pngChunk('IEND', new Uint8Array(0));

    const out = new Uint8Array(sig.length + cIhdr.length + cIdat.length + cIend.length);
    let offset = 0;
    for (const part of [sig, cIhdr, cIdat, cIend]) {
      out.set(part, offset);
      offset += part.length;
    }
    return out;
  }
}

function dist3(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function catmullRom3(
  p0: readonly [number, number, number],
  p1: readonly [number, number, number],
  p2: readonly [number, number, number],
  p3: readonly [number, number, number],
  t: number,
): [number, number, number] {
  const t2 = t * t;
  const t3 = t2 * t;
  const eval1 = (v0: number, v1: number, v2: number, v3: number) =>
    0.5 *
    (2 * v1 +
      (-v0 + v2) * t +
      (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2 +
      (-v0 + 3 * v1 - 3 * v2 + v3) * t3);
  return [
    eval1(p0[0], p1[0], p2[0], p3[0]),
    eval1(p0[1], p1[1], p2[1], p3[1]),
    eval1(p0[2], p1[2], p2[2], p3[2]),
  ];
}

function plddtToRgb(plddt: number): [number, number, number] {
  if (plddt >= 90) return [0, 83, 214];
  if (plddt >= 70) return [69, 179, 230];
  if (plddt >= 50) return [245, 200, 42];
  return [240, 120, 62];
}

function spectrumRgb(t: number): [number, number, number] {
  // Smooth N->C teal -> azure -> royal blue -> violet -> coral
  const stops: Array<[number, number, number]> = [
    [13, 148, 136],
    [14, 165, 233],
    [59, 130, 246],
    [139, 92, 246],
    [236, 72, 153],
  ];
  const u = clamp(t, 0, 1) * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(u));
  const f = u - idx;
  return [
    Math.round(lerp(stops[idx][0], stops[idx + 1][0], f)),
    Math.round(lerp(stops[idx][1], stops[idx + 1][1], f)),
    Math.round(lerp(stops[idx][2], stops[idx + 1][2], f)),
  ];
}

/** 3x3 symmetric Jacobi eigendecomposition returning orthonormal rows sorted by descending variance. */
function jacobiPrincipalAxes(
  c00: number,
  c01: number,
  c02: number,
  c11: number,
  c12: number,
  c22: number,
): [[number, number, number], [number, number, number], [number, number, number]] {
  const A = [
    [c00, c01, c02],
    [c01, c11, c12],
    [c02, c12, c22],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iter = 0; iter < 16; iter++) {
    let p = 0;
    let q = 1;
    let maxVal = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > maxVal) {
      maxVal = Math.abs(A[0][2]);
      p = 0;
      q = 2;
    }
    if (Math.abs(A[1][2]) > maxVal) {
      maxVal = Math.abs(A[1][2]);
      p = 1;
      q = 2;
    }
    if (maxVal < 1e-8) break;
    const diff = A[q][q] - A[p][p];
    const t =
      Math.abs(A[p][q]) < Math.abs(diff) * 1e-18
        ? A[p][q] / diff
        : (() => {
            const phi = diff / (2 * A[p][q]);
            return 1 / (Math.abs(phi) + Math.sqrt(phi * phi + 1)) * (phi < 0 ? -1 : 1);
          })();
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = 0;
    A[q][p] = 0;
    const r = 3 - p - q;
    const arp = A[r][p];
    const arq = A[r][q];
    A[r][p] = A[p][r] = c * arp - s * arq;
    A[r][q] = A[q][r] = s * arp + c * arq;
    for (let i = 0; i < 3; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }
  const order = [0, 1, 2].sort((a, b) => A[b][b] - A[a][a]);
  const row = (col: number): [number, number, number] => [V[0][col], V[1][col], V[2][col]];
  return [row(order[0]), row(order[1]), row(order[2])];
}

async function deflateZlib(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream('deflate');
  const writer = cs.writable.getWriter();
  const bufPromise = new Response(cs.readable).arrayBuffer();
  bufPromise.catch(() => {});
  const writePromise = (async () => {
    await writer.write(data.slice());
    await writer.close();
  })();
  writePromise.catch(() => {});
  const [buf] = await Promise.all([bufPromise, writePromise]);
  return new Uint8Array(buf);
}

const CRC_TABLE = (() => {
  const tbl = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    tbl[n] = c >>> 0;
  }
  return tbl;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crc = crc32(out.subarray(4, 8 + data.length));
  dv.setUint32(8 + data.length, crc, false);
  return out;
}
