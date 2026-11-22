import type { StructureArtifact } from './types';

export type Dyna1InputModeParam = 'auto' | 'seq+struct' | 'seq-only' | 'struct-only';
export type Dyna1ResolvedMode = 'seq+struct' | 'seq-only' | 'struct-only';

export function normalizeDyna1InputMode(raw: unknown): Dyna1InputModeParam {
  const val = String(raw ?? 'auto').trim().toLowerCase();
  if (val === 'seq+struct' || val === 'seq-only' || val === 'struct-only') return val;
  return 'auto';
}

export interface ExtractedDyna1Backbone {
  /** Flat Float32Array(L * 9) in [N, CA, C] × [x, y, z] Å order; NaN for missing atoms. */
  coords: Float32Array;
  /** Number of residues where N, CA, and C are all observed and finite. */
  validResidueCount: number;
  length: number;
}

/**
 * Extract ESM-3 StructureTokenEncoder `[N, CA, C]` coordinates (`Float32Array(L * 9)`)
 * from a StructureArtifact's `backbone4Coords` (`[N, CA, C, O]`, 12 floats/residue).
 * Missing atoms (per `atomPresence` or non-finite values) are encoded strictly as `NaN`
 * so ESM-3's local-frame mask handles them identically to upstream `ProteinChain`.
 */
export function extractDyna1Backbone3Coords(
  struct: Pick<StructureArtifact, 'length' | 'backbone4Coords' | 'atomPresence' | 'atomGranularity'> | undefined,
): ExtractedDyna1Backbone | null {
  if (!struct || struct.length < 1) return null;
  if (struct.atomGranularity === 'ca-only') return null;
  const bb = struct.backbone4Coords;
  if (!bb || bb.length < struct.length * 12) return null;

  const L = struct.length;
  const coords = new Float32Array(L * 9);
  const presence = struct.atomPresence;
  let validResidueCount = 0;

  for (let r = 0; r < L; r++) {
    const src = r * 12;
    const dst = r * 9;
    const pres = presence && presence.length === L ? presence[r] : undefined;

    const nx = bb[src + 0];
    const ny = bb[src + 1];
    const nz = bb[src + 2];
    const cax = bb[src + 3];
    const cay = bb[src + 4];
    const caz = bb[src + 5];
    const cx = bb[src + 6];
    const cy = bb[src + 7];
    const cz = bb[src + 8];

    const nFinite = Number.isFinite(nx) && Number.isFinite(ny) && Number.isFinite(nz);
    const caFinite = Number.isFinite(cax) && Number.isFinite(cay) && Number.isFinite(caz);
    const cFinite = Number.isFinite(cx) && Number.isFinite(cy) && Number.isFinite(cz);

    const hasN = (pres ? pres.hasN : true) && nFinite;
    const hasCA = (pres ? pres.hasCA : true) && caFinite;
    const hasC = (pres ? pres.hasC : true) && cFinite;

    // Guard against uninitialized all-zero synthetic N/C slots when atomPresence wasn't populated
    const zeroNC =
      !pres &&
      nx === 0 && ny === 0 && nz === 0 &&
      cx === 0 && cy === 0 && cz === 0 &&
      !(cax === 0 && cay === 0 && caz === 0);

    if (hasN && !zeroNC) {
      coords[dst + 0] = nx;
      coords[dst + 1] = ny;
      coords[dst + 2] = nz;
    } else {
      coords[dst + 0] = Number.NaN;
      coords[dst + 1] = Number.NaN;
      coords[dst + 2] = Number.NaN;
    }

    if (hasCA) {
      coords[dst + 3] = cax;
      coords[dst + 4] = cay;
      coords[dst + 5] = caz;
    } else {
      coords[dst + 3] = Number.NaN;
      coords[dst + 4] = Number.NaN;
      coords[dst + 5] = Number.NaN;
    }

    if (hasC && !zeroNC) {
      coords[dst + 6] = cx;
      coords[dst + 7] = cy;
      coords[dst + 8] = cz;
    } else {
      coords[dst + 6] = Number.NaN;
      coords[dst + 7] = Number.NaN;
      coords[dst + 8] = Number.NaN;
    }

    if (hasN && hasCA && hasC && !zeroNC) {
      validResidueCount++;
    }
  }

  return { coords, validResidueCount, length: L };
}

/**
 * Resolve the effective Dyna-1 conditioning mode for a candidate given the user's
 * requested `inputMode` and whether matching full-backbone `(N, CA, C)` coordinates exist.
 */
export function resolveDyna1Mode(
  requested: Dyna1InputModeParam,
  seqLength: number,
  backbone: ExtractedDyna1Backbone | null,
): Dyna1ResolvedMode {
  const hasMatchingBackbone =
    backbone !== null &&
    backbone.validResidueCount >= 3 &&
    (seqLength === 0 || backbone.length === seqLength);

  if (requested === 'seq-only') return 'seq-only';
  if (requested === 'seq+struct') {
    if (!hasMatchingBackbone) {
      throw new Error(
        'Dyna-1 "Sequence + Structure" mode requires a connected 3D structure with matching residue count and full backbone (N, Cα, C) atoms.',
      );
    }
    return 'seq+struct';
  }
  if (requested === 'struct-only') {
    if (!backbone || backbone.validResidueCount < 3) {
      throw new Error(
        'Dyna-1 "Structure-only" mode requires a connected 3D structure with full backbone (N, Cα, C) atoms.',
      );
    }
    return 'struct-only';
  }
  // 'auto': use full sequence + 3D structure conditioning whenever matching backbone is present
  return hasMatchingBackbone ? 'seq+struct' : 'seq-only';
}
