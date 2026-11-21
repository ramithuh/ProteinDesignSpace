// Pure-TS Dunbrack-lab ipSAE (interaction Prediction Score from Aligned Errors)
// Reference: Roland L. Dunbrack, Jr. (2025), "What's wrong with AlphaFold's ipTM score and how to fix it"
// https://www.biorxiv.org/content/10.1101/2025.02.10.637595

export interface IpsaeResult {
  /** Symmetrized maximum directional ipSAE across distinct chain pairs (canonical Dunbrack ipSAE in [0, 1]). */
  ipsae: number;
  /** Maximum directional ipSAE across distinct chain pairs. */
  ipsaeMax: number;
  /** Minimum directional ipSAE across distinct chain pairs. */
  ipsaeMin: number;
  /** PAE cutoff in Å used to identify aligned cross-chain interface residues (default 10 Å). */
  paeCutoff: number;
  /** Count of unique residues participating in at least one cross-chain pair with PAE < paeCutoff. */
  interfaceResidues: number;
  /** Per-direction chain-pair ipSAE scores keyed by `${chain1}->${chain2}`. */
  pairScores: Record<string, number>;
}

/**
 * Compute Dunbrack (2025) / Yang & Skolnick d0(n0) normalization scale in Å for protein interfaces:
 * d0(n0) = max(1.0, 1.24 * (max(n0, 27) - 15)^(1/3) - 1.8)
 */
export function computeIpsaeD0(n0: number): number {
  const effectiveN = Math.max(n0, 27);
  const d0 = 1.24 * Math.cbrt(effectiveN - 15) - 1.8;
  return Math.max(1.0, d0);
}

/**
 * Extract the leading `proteinLen × proteinLen` protein submatrix from a square PAE matrix.
 * Returns the original `Float32Array` without copying when `pae.length === proteinLen * proteinLen`,
 * or slices the top-left `proteinLen × proteinLen` block when AF3-class models append non-protein tokens.
 */
export function sliceProteinPae(
  pae: Float32Array | ArrayLike<number>,
  proteinLen: number,
): Float32Array | null {
  if (!Number.isInteger(proteinLen) || proteinLen <= 0) return null;
  if (pae.length < proteinLen * proteinLen) return null;
  const paeDim = Math.round(Math.sqrt(pae.length));
  if (paeDim < proteinLen || paeDim * paeDim !== pae.length) return null;

  if (paeDim === proteinLen) {
    return pae instanceof Float32Array ? pae : Float32Array.from(pae);
  }
  const sliced = new Float32Array(proteinLen * proteinLen);
  for (let i = 0; i < proteinLen; i++) {
    const srcRow = i * paeDim;
    const dstRow = i * proteinLen;
    for (let j = 0; j < proteinLen; j++) {
      sliced[dstRow + j] = pae[srcRow + j];
    }
  }
  return sliced;
}

/**
 * Compute Dunbrack-lab `ipSAE` from a flat row-major `L × L` protein PAE matrix (in Å)
 * and per-residue `chainIds` of length `L`.
 *
 * Returns `null` when `chainIds` has fewer than 2 distinct chains, `pae.length !== L * L`,
 * or `paeCutoff` is not a positive finite number.
 */
export function computeIpsae(
  pae: Float32Array | ArrayLike<number>,
  chainIds: readonly string[],
  paeCutoff = 10,
): IpsaeResult | null {
  const L = chainIds.length;
  if (L === 0 || pae.length !== L * L) return null;
  if (!Number.isFinite(paeCutoff) || paeCutoff <= 0) return null;

  const chainOrder: string[] = [];
  const chainIndices = new Map<string, number[]>();
  for (let i = 0; i < L; i++) {
    const cid = chainIds[i] || 'A';
    let list = chainIndices.get(cid);
    if (!list) {
      list = [];
      chainIndices.set(cid, list);
      chainOrder.push(cid);
    }
    list.push(i);
  }

  if (chainOrder.length < 2) return null;

  const pairScores: Record<string, number> = {};
  const interfaceMask = new Uint8Array(L);
  let ipsaeMax = -Infinity;
  let ipsaeMin = Infinity;

  for (let a = 0; a < chainOrder.length; a++) {
    const c1 = chainOrder[a];
    const idx1 = chainIndices.get(c1)!;
    for (let b = 0; b < chainOrder.length; b++) {
      if (a === b) continue;
      const c2 = chainOrder[b];
      const idx2 = chainIndices.get(c2)!;

      let bestDir = 0;
      for (let ii = 0; ii < idx1.length; ii++) {
        const i = idx1[ii];
        const rowOffset = i * L;

        let n0 = 0;
        for (let jj = 0; jj < idx2.length; jj++) {
          const j = idx2[jj];
          const v = pae[rowOffset + j];
          if (Number.isFinite(v) && v < paeCutoff) {
            n0++;
          }
        }
        if (n0 === 0) continue;

        interfaceMask[i] = 1;
        const d0 = computeIpsaeD0(n0);
        const invD0Sq = 1 / (d0 * d0);
        let sum = 0;
        for (let jj = 0; jj < idx2.length; jj++) {
          const j = idx2[jj];
          const v = pae[rowOffset + j];
          if (Number.isFinite(v) && v < paeCutoff) {
            interfaceMask[j] = 1;
            sum += 1 / (1 + v * v * invD0Sq);
          }
        }
        const scoreI = sum / n0;
        if (scoreI > bestDir) {
          bestDir = scoreI;
        }
      }

      pairScores[`${c1}->${c2}`] = bestDir;
      if (bestDir > ipsaeMax) ipsaeMax = bestDir;
      if (bestDir < ipsaeMin) ipsaeMin = bestDir;
    }
  }

  let interfaceResidues = 0;
  for (let i = 0; i < L; i++) {
    if (interfaceMask[i]) interfaceResidues++;
  }

  const finalMax = Number.isFinite(ipsaeMax) ? ipsaeMax : 0;
  const finalMin = Number.isFinite(ipsaeMin) ? ipsaeMin : 0;

  return {
    ipsae: finalMax,
    ipsaeMax: finalMax,
    ipsaeMin: finalMin,
    paeCutoff,
    interfaceResidues,
    pairScores,
  };
}
