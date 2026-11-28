/**
 * Coordinate and graph featurizer for the ArtiosPotts ports — the JS mirror of
 * `artios.data.features.featurizer.ContextGenerator` plus the geometric part of
 * `artios.model.modules.mpnn_features.MPNNFeaturizer`, with no RDKit, no CCD and
 * no structure-parsing dependencies.
 *
 * Input is a plain "spec" object:
 *
 * ```js
 * {
 *   name: 'example',
 *   chains: [
 *     { id: 'A', kind: 'protein', residues: [
 *         { resName: 'THR', resSeq: 1, N: [x,y,z], CA: [...], C: [...], O: [...] },
 *     ] },
 *     { id: 'B', kind: 'ligand', residues: [
 *         { resName: 'NAG', resSeq: 1, atoms: [
 *             { name: 'C1', element: 'C', charge: 0, coord: [x,y,z] },
 *         ] },
 *     ] },
 *   ],
 * }
 * ```
 *
 * Tokenization follows AF3 SI 2.6 as artios implements it for this input domain:
 * one token per standard protein residue (centre atom CA; five backbone slots
 * N/CA/C/O plus the ProteinMPNN virtual CB) and one token per context heavy atom
 * (centre atom itself; backbone slot 0 only). `tools/potts/dump_oracle.py`
 * implements exactly these rules in Python, and the CPU tests assert the two
 * agree tensor for tensor.
 */

import {
  BACKBONE_PAIRS,
  ELEMENT_Z,
  INDEX_TO_ONE_LETTER,
  NUM_BACKBONE_ATOMS,
  NUM_ELEMENTS,
  NUM_RESIDUE_TYPES,
  ONE_LETTER_TO_INDEX,
  PROTEIN_BACKBONE,
  PROTEIN_THREE,
  RESTYPE_INDEX,
  UNK_INDEX,
} from './constants.js';

/**
 * ProteinMPNN ideal-geometry virtual Cβ, as `artios.utils.geometry.compute_virtual_cb`.
 * Writes into `out` at `[ox, ox+1, ox+2]`.
 */
export function virtualCb(n, ca, c, out, ox = 0) {
  const bx = ca[0] - n[0];
  const by = ca[1] - n[1];
  const bz = ca[2] - n[2];
  const cx = c[0] - ca[0];
  const cy = c[1] - ca[1];
  const cz = c[2] - ca[2];
  // cross(n_ca, ca_c)
  const ax = by * cz - bz * cy;
  const ay = bz * cx - bx * cz;
  const az = bx * cy - by * cx;
  out[ox] = ca[0] + 0.568 * bx - 0.54068 * cx - 0.5827 * ax;
  out[ox + 1] = ca[1] + 0.568 * by - 0.54068 * cy - 0.5827 * ay;
  out[ox + 2] = ca[2] + 0.568 * bz - 0.54068 * cz - 0.5827 * az;
}

function elementChannel(symbol) {
  const z = ELEMENT_Z[String(symbol).toUpperCase()];
  if (z === undefined) throw new Error(`unknown element ${JSON.stringify(symbol)}`);
  return z - 1; // get_all_elems() lists Z = 1..118 first, so the channel is Z - 1.
}

/**
 * Featurize a spec into the token- and atom-level tensors the MPNN reads.
 *
 * Returned arrays are flat and row major; `shape` comments give the logical shape.
 * @returns {object} context
 */
export function featurize(spec) {
  if (!spec || !Array.isArray(spec.chains)) {
    throw new Error('featurize needs a spec with a `chains` array');
  }

  const tokens = [];
  const atoms = [];
  const asymKeys = [];
  const asymEntityKey = [];
  const asymIndex = new Map();

  const asymIdFor = (key, entityKey) => {
    let found = asymIndex.get(key);
    if (found === undefined) {
      found = asymKeys.length;
      asymIndex.set(key, found);
      asymKeys.push(key);
      asymEntityKey.push(entityKey);
    }
    return found;
  };

  for (const chain of spec.chains) {
    const chainId = String(chain.id);
    if (chain.kind === 'protein') {
      const entityKey = `poly:${chain.residues.map((r) => r.resName).join(',')}`;
      const asym = asymIdFor(`chain:${chainId}`, entityKey);
      for (const residue of chain.residues) {
        const tokenIdx = tokens.length;
        const coords = new Float32Array(NUM_BACKBONE_ATOMS * 3);
        const mask = new Uint8Array(NUM_BACKBONE_ATOMS);
        for (let slot = 0; slot < PROTEIN_BACKBONE.length; slot += 1) {
          const atomName = PROTEIN_BACKBONE[slot];
          const value = residue[atomName];
          if (!value) continue;
          coords[slot * 3] = value[0];
          coords[slot * 3 + 1] = value[1];
          coords[slot * 3 + 2] = value[2];
          mask[slot] = 1;
          atoms.push({
            token: tokenIdx,
            // Backbone element symbols: N, CA -> C, C -> C, O.
            element: atomName === 'CA' ? 'C' : atomName[0],
            charge: 0,
            refMask: 1,
            isLigand: 0,
          });
        }
        // Slots 0/1/2 are N/CA/C; read them back out of `coords` so the virtual
        // CB is built from float32-rounded inputs, as the PyTorch path does.
        if (mask[0] && mask[1] && mask[2]) {
          virtualCb(coords.subarray(0, 3), coords.subarray(3, 6), coords.subarray(6, 9), coords, 12);
          mask[4] = 1;
        }
        if (!mask[1]) {
          throw new Error(
            `residue ${chainId}${residue.resSeq} has no CA, so it has no centre atom`,
          );
        }
        tokens.push({
          restype: RESTYPE_INDEX[residue.resName] ?? UNK_INDEX,
          residueIndex: Number(residue.resSeq),
          asymId: asym,
          coords,
          mask,
          centre: [coords[3], coords[4], coords[5]],
          isProtein: 1,
          isLigand: 0,
          meta: {
            kind: 'protein',
            chain: chainId,
            resSeq: Number(residue.resSeq),
            resName: residue.resName,
          },
        });
      }
    } else if (chain.kind === 'ligand') {
      for (const residue of chain.residues) {
        // Each context residue is its own label chain, as in artios' mmCIF path.
        const entityKey = `ligand:${residue.resName}`;
        const asym = asymIdFor(
          `ligand:${chainId}:${residue.resSeq}:${residue.resName}`,
          entityKey,
        );
        for (const atom of residue.atoms) {
          const tokenIdx = tokens.length;
          const coords = new Float32Array(NUM_BACKBONE_ATOMS * 3);
          const mask = new Uint8Array(NUM_BACKBONE_ATOMS);
          coords[0] = atom.coord[0];
          coords[1] = atom.coord[1];
          coords[2] = atom.coord[2];
          mask[0] = 1;
          atoms.push({
            token: tokenIdx,
            element: atom.element,
            charge: atom.charge ?? 0,
            refMask: atom.refMask ?? 1,
            isLigand: 1,
          });
          tokens.push({
            restype: UNK_INDEX,
            residueIndex: Number(residue.resSeq),
            asymId: asym,
            coords,
            mask,
            centre: [coords[0], coords[1], coords[2]],
            isProtein: 0,
            isLigand: 1,
            meta: {
              kind: 'ligand',
              chain: chainId,
              resSeq: Number(residue.resSeq),
              resName: residue.resName,
              atomName: atom.name,
            },
          });
        }
      }
    } else {
      throw new Error(`unsupported chain kind ${JSON.stringify(chain.kind)}`);
    }
  }

  // entity_id: entity key in first-appearance order; sym_id: per-entity copy
  // counter in asym order (artios.data.structure.annotation.EntityIds).
  const entityIndex = new Map();
  const symCounter = new Map();
  const entityOfAsym = [];
  const symOfAsym = [];
  for (const key of asymEntityKey) {
    if (!entityIndex.has(key)) entityIndex.set(key, entityIndex.size);
    entityOfAsym.push(entityIndex.get(key));
    const seen = symCounter.get(key) ?? 0;
    symOfAsym.push(seen);
    symCounter.set(key, seen + 1);
  }

  const nToken = tokens.length;
  const nAtom = atoms.length;
  if (nToken === 0) throw new Error('spec produced no tokens');

  const tokenIndex = new Int32Array(nToken);
  const residueIndex = new Int32Array(nToken);
  const asymId = new Int32Array(nToken);
  const entityId = new Int32Array(nToken);
  const symId = new Int32Array(nToken);
  const restypeIndex = new Int32Array(nToken);
  const restype = new Int32Array(nToken * NUM_RESIDUE_TYPES); // [N, 32] one-hot
  const tokenMask = new Uint8Array(nToken).fill(1);
  const isProteinToken = new Uint8Array(nToken);
  const isLigandToken = new Uint8Array(nToken);
  const isDnaToken = new Uint8Array(nToken);
  const isRnaToken = new Uint8Array(nToken);
  const backboneCoordinate = new Float32Array(nToken * NUM_BACKBONE_ATOMS * 3);
  const backboneMask = new Uint8Array(nToken * NUM_BACKBONE_ATOMS);
  const centreCoordinate = new Float32Array(nToken * 3);
  const residues = new Array(nToken);

  for (let i = 0; i < nToken; i += 1) {
    const token = tokens[i];
    tokenIndex[i] = i;
    residueIndex[i] = token.residueIndex;
    asymId[i] = token.asymId;
    entityId[i] = entityOfAsym[token.asymId];
    symId[i] = symOfAsym[token.asymId];
    restypeIndex[i] = token.restype;
    restype[i * NUM_RESIDUE_TYPES + token.restype] = 1;
    isProteinToken[i] = token.isProtein;
    isLigandToken[i] = token.isLigand;
    backboneCoordinate.set(token.coords, i * NUM_BACKBONE_ATOMS * 3);
    backboneMask.set(token.mask, i * NUM_BACKBONE_ATOMS);
    centreCoordinate[i * 3] = token.centre[0];
    centreCoordinate[i * 3 + 1] = token.centre[1];
    centreCoordinate[i * 3 + 2] = token.centre[2];
    residues[i] = token.meta;
  }

  const atomToTokenIdx = new Int32Array(nAtom);
  const atomMask = new Uint8Array(nAtom).fill(1);
  const isLigandAtom = new Uint8Array(nAtom);
  const isMotifRelatedAtom = new Uint8Array(nAtom);
  const refCharge = new Int32Array(nAtom);
  const refMask = new Int32Array(nAtom);
  const refElementIndex = new Int32Array(nAtom);
  for (let a = 0; a < nAtom; a += 1) {
    const atom = atoms[a];
    atomToTokenIdx[a] = atom.token;
    isLigandAtom[a] = atom.isLigand;
    refCharge[a] = atom.charge;
    refMask[a] = atom.refMask;
    refElementIndex[a] = elementChannel(atom.element);
  }

  return {
    name: spec.name ?? 'unnamed',
    nToken,
    nAtom,
    numResidueTypes: NUM_RESIDUE_TYPES,
    numElements: NUM_ELEMENTS,
    tokenIndex,
    residueIndex,
    asymId,
    entityId,
    symId,
    restype,
    restypeIndex,
    tokenMask,
    isProteinToken,
    isLigandToken,
    isDnaToken,
    isRnaToken,
    backboneCoordinate,
    backboneMask,
    centreCoordinate,
    atomToTokenIdx,
    atomMask,
    isLigandAtom,
    isMotifRelatedAtom,
    refCharge,
    refMask,
    refElementIndex,
    residues,
  };
}

/** Dense `ref_element` one-hot `[nAtom, 128]`, if a caller needs it explicitly. */
export function refElementOneHot(context) {
  const out = new Float32Array(context.nAtom * NUM_ELEMENTS);
  for (let a = 0; a < context.nAtom; a += 1) {
    out[a * NUM_ELEMENTS + context.refElementIndex[a]] = 1;
  }
  return out;
}

/**
 * k-nearest-neighbour graph over token centre atoms.
 *
 * Mirrors `MPNNFeaturizer._compute_neighbour_graph`: masked-out tokens are pushed
 * to the far end by adding each row's maximum distance, then the `k` smallest
 * entries per row are taken in ascending order.
 *
 * Exact distance ties break on node index here. `torch.topk(..., largest=False)`
 * does not define its order within a tie group — on CPU it is an artifact of the
 * selection kernel — so a perfectly symmetric structure (an exact helix, where
 * token i sits the same distance from i-m and i+m) can legitimately give a
 * different, equally valid neighbour list in the two implementations. Use
 * `findDistanceTies` to check whether a structure is affected.
 */
export function buildNeighbourGraph(context, numNeighbours = 32) {
  const { nToken, centreCoordinate, tokenMask } = context;
  const k = Math.min(numNeighbours, nToken);
  const dist = new Float64Array(nToken * nToken);

  for (let i = 0; i < nToken; i += 1) {
    const xi = centreCoordinate[i * 3];
    const yi = centreCoordinate[i * 3 + 1];
    const zi = centreCoordinate[i * 3 + 2];
    for (let j = 0; j < nToken; j += 1) {
      const valid = tokenMask[i] && tokenMask[j];
      if (!valid) continue; // pairwise_dist * mask_2d
      const dx = xi - centreCoordinate[j * 3];
      const dy = yi - centreCoordinate[j * 3 + 1];
      const dz = zi - centreCoordinate[j * 3 + 2];
      dist[i * nToken + j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
  }

  const edgeIndex = new Int32Array(nToken * k);
  const neighbourDistance = new Float32Array(nToken * k);
  const adjusted = new Float64Array(nToken);
  const order = new Array(nToken);
  for (let i = 0; i < nToken; i += 1) {
    let dMax = -Infinity;
    for (let j = 0; j < nToken; j += 1) {
      if (dist[i * nToken + j] > dMax) dMax = dist[i * nToken + j];
    }
    for (let j = 0; j < nToken; j += 1) {
      const valid = tokenMask[i] && tokenMask[j] ? 1 : 0;
      adjusted[j] = dist[i * nToken + j] + (1 - valid) * dMax;
      order[j] = j;
    }
    order.sort((a, b) => (adjusted[a] - adjusted[b]) || (a - b));
    for (let s = 0; s < k; s += 1) {
      edgeIndex[i * k + s] = order[s];
      neighbourDistance[i * k + s] = adjusted[order[s]];
    }
  }

  // MPNN.encode: edge_mask = token_mask_i & token_mask_j.
  const edgeMask = new Uint8Array(nToken * k);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      edgeMask[i * k + s] = tokenMask[i] && tokenMask[edgeIndex[i * k + s]] ? 1 : 0;
    }
  }

  return { k, edgeIndex, edgeMask, neighbourDistance };
}

/**
 * Report exact float32 centre-distance ties, which make the kNN neighbour list
 * order ambiguous (see `buildNeighbourGraph`).
 *
 * @returns {{node: number, distance: number, neighbours: number[]}[]}
 */
export function findDistanceTies(context) {
  const { nToken, centreCoordinate, tokenMask } = context;
  const scratch = new Float32Array(1);
  const ties = [];
  for (let i = 0; i < nToken; i += 1) {
    if (!tokenMask[i]) continue;
    const groups = new Map();
    for (let j = 0; j < nToken; j += 1) {
      if (!tokenMask[j] || j === i) continue;
      const dx = centreCoordinate[i * 3] - centreCoordinate[j * 3];
      const dy = centreCoordinate[i * 3 + 1] - centreCoordinate[j * 3 + 1];
      const dz = centreCoordinate[i * 3 + 2] - centreCoordinate[j * 3 + 2];
      scratch[0] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const key = scratch[0];
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(j);
    }
    for (const [distance, neighbours] of groups) {
      if (neighbours.length > 1) ties.push({ node: i, distance, neighbours });
    }
  }
  return ties;
}

/**
 * RBF-encoded distances between all 25 ordered backbone-atom pairs of each edge.
 *
 * Mirrors `MPNNFeaturizer._compute_backbone_distance_features`; the flattened
 * pair index is `selfSlot * 5 + neighbourSlot` and the output is
 * `[nToken, k, 25 * numRbf]`.
 */
export function backboneDistanceRbf(context, graph, options = {}) {
  const {
    numRbf = 16,
    minDistance = 2.0,
    maxDistance = 22.0,
    // `GaussianRBF` registers its centres and gamma as buffers, so a checkpoint
    // can supply them verbatim instead of having them recomputed here.
    centers: givenCenters = null,
    gamma: givenGamma = null,
  } = options;
  const { nToken, backboneCoordinate, backboneMask } = context;
  const { k, edgeIndex } = graph;

  let centers;
  if (givenCenters) {
    if (givenCenters.length !== numRbf) {
      throw new Error(`got ${givenCenters.length} RBF centres, expected ${numRbf}`);
    }
    centers = givenCenters;
  } else {
    centers = new Float64Array(numRbf);
    for (let m = 0; m < numRbf; m += 1) {
      centers[m] = numRbf === 1
        ? minDistance
        : minDistance + ((maxDistance - minDistance) * m) / (numRbf - 1);
    }
  }
  const spacing = (maxDistance - minDistance) / (numRbf - 1);
  const gamma = givenGamma ?? 1 / (spacing * spacing);

  const featureDim = BACKBONE_PAIRS * numRbf;
  const out = new Float32Array(nToken * k * featureDim);

  for (let i = 0; i < nToken; i += 1) {
    const iBase = i * NUM_BACKBONE_ATOMS;
    for (let s = 0; s < k; s += 1) {
      const j = edgeIndex[i * k + s];
      const jBase = j * NUM_BACKBONE_ATOMS;
      const edgeBase = (i * k + s) * featureDim;
      for (let r = 0; r < NUM_BACKBONE_ATOMS; r += 1) {
        if (!backboneMask[iBase + r]) continue; // masked pairs stay exactly zero
        const rx = backboneCoordinate[(iBase + r) * 3];
        const ry = backboneCoordinate[(iBase + r) * 3 + 1];
        const rz = backboneCoordinate[(iBase + r) * 3 + 2];
        for (let d = 0; d < NUM_BACKBONE_ATOMS; d += 1) {
          if (!backboneMask[jBase + d]) continue;
          const dx = rx - backboneCoordinate[(jBase + d) * 3];
          const dy = ry - backboneCoordinate[(jBase + d) * 3 + 1];
          const dz = rz - backboneCoordinate[(jBase + d) * 3 + 2];
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const pairBase = edgeBase + (r * NUM_BACKBONE_ATOMS + d) * numRbf;
          for (let m = 0; m < numRbf; m += 1) {
            const delta = distance - centers[m];
            out[pairBase + m] = Math.exp(-gamma * delta * delta);
          }
        }
      }
    }
  }
  return { data: out, featureDim, centers, gamma };
}

/**
 * One-hot relative position features per edge, `[nToken, k, 4*rMax + 2*sMax + 7]`.
 *
 * Mirrors `RelativePositionEncoding` (AF3 Algorithm 3) gathered at the kNN edges,
 * i.e. `gather_edges(relpos, edge_index)`. Channel layout:
 * relative residue (2*rMax+2), relative token (2*rMax+2), same entity (1),
 * relative chain (2*sMax+2).
 */
export function relativePositionFeatures(context, graph, options = {}) {
  const { rMax = 32, sMax = 2 } = options;
  const { nToken, asymId, residueIndex, entityId, symId, tokenIndex } = context;
  const { k, edgeIndex } = graph;

  const residueBins = 2 * rMax + 2;
  const chainBins = 2 * sMax + 2;
  const featureDim = 2 * residueBins + 1 + chainBins;
  const differentChainIdx = 2 * rMax + 1;
  const differentEntityIdx = 2 * sMax + 1;
  const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  const out = new Float32Array(nToken * k * featureDim);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const j = edgeIndex[i * k + s];
      const base = (i * k + s) * featureDim;
      const sameChain = asymId[i] === asymId[j];
      const sameResidue = residueIndex[i] === residueIndex[j];
      const sameEntity = entityId[i] === entityId[j];

      const dResidue = sameChain
        ? clip(residueIndex[i] - residueIndex[j] + rMax, 0, 2 * rMax)
        : differentChainIdx;
      out[base + dResidue] = 1;

      const dToken = sameChain && sameResidue
        ? clip(tokenIndex[i] - tokenIndex[j] + rMax, 0, 2 * rMax)
        : differentChainIdx;
      out[base + residueBins + dToken] = 1;

      if (sameEntity) out[base + 2 * residueBins] = 1;

      const dChain = sameEntity
        ? clip(symId[i] - symId[j] + sMax, 0, 2 * sMax)
        : differentEntityIdx;
      out[base + 2 * residueBins + 1 + dChain] = 1;
    }
  }
  return { data: out, featureDim };
}

/**
 * Non-ligand tokens with a backbone atom within `distanceCutoff` of a context
 * token's centre — `artios.model.utils.build_backbone_ligand_proximity_mask`.
 */
export function buildLigandProximityMask(context, distanceCutoff) {
  const { nToken, backboneCoordinate, backboneMask, centreCoordinate, isLigandToken, tokenMask } =
    context;
  const near = new Uint8Array(nToken);
  const ligands = [];
  for (let j = 0; j < nToken; j += 1) {
    if (isLigandToken[j] && tokenMask[j]) ligands.push(j);
  }
  if (ligands.length === 0) return near;

  for (let i = 0; i < nToken; i += 1) {
    if (isLigandToken[i] || !tokenMask[i]) continue;
    let isNear = false;
    for (let slot = 0; slot < NUM_BACKBONE_ATOMS && !isNear; slot += 1) {
      if (!backboneMask[i * NUM_BACKBONE_ATOMS + slot]) continue;
      const bx = backboneCoordinate[(i * NUM_BACKBONE_ATOMS + slot) * 3];
      const by = backboneCoordinate[(i * NUM_BACKBONE_ATOMS + slot) * 3 + 1];
      const bz = backboneCoordinate[(i * NUM_BACKBONE_ATOMS + slot) * 3 + 2];
      for (const j of ligands) {
        const dx = bx - centreCoordinate[j * 3];
        const dy = by - centreCoordinate[j * 3 + 1];
        const dz = bz - centreCoordinate[j * 3 + 2];
        if (Math.sqrt(dx * dx + dy * dy + dz * dz) <= distanceCutoff) {
          isNear = true;
          break;
        }
      }
    }
    near[i] = isNear ? 1 : 0;
  }
  return near;
}

/**
 * A copy of `context` with every ligand influence removed, for the
 * classifier-free-guidance pathway — `FullContext.without_ligand`.
 *
 * Ligand tokens stay in the tensors but drop out of `tokenMask` (so out of the
 * kNN graph, the node features and every edge mask) and their centre coordinates
 * are zeroed. `backboneCoordinate` and `restype` are left alone, as in the
 * reference.
 */
export function withoutLigand(context) {
  const { nToken, nAtom } = context;
  const tokenMask = new Uint8Array(nToken);
  const centreCoordinate = Float32Array.from(context.centreCoordinate);
  for (let i = 0; i < nToken; i += 1) {
    tokenMask[i] = context.tokenMask[i] && !context.isLigandToken[i] ? 1 : 0;
    if (context.isLigandToken[i]) {
      centreCoordinate[i * 3] = 0;
      centreCoordinate[i * 3 + 1] = 0;
      centreCoordinate[i * 3 + 2] = 0;
    }
  }
  const atomMask = new Uint8Array(nAtom);
  for (let a = 0; a < nAtom; a += 1) {
    atomMask[a] = context.atomMask[a] && !context.isLigandAtom[a] ? 1 : 0;
  }
  return {
    ...context,
    tokenMask,
    centreCoordinate,
    atomMask,
    isLigandToken: new Uint8Array(nToken),
    isLigandAtom: new Uint8Array(nAtom),
  };
}

/** Per-chain one-letter sequences of the protein tokens, in token order. */
export function sequencesByChain(context, sequence = context.restypeIndex) {
  const out = new Map();
  for (let i = 0; i < context.nToken; i += 1) {
    const meta = context.residues[i];
    if (meta.kind !== 'protein') continue;
    if (!out.has(meta.chain)) out.set(meta.chain, []);
    out.get(meta.chain).push(INDEX_TO_ONE_LETTER[sequence[i]] ?? 'X');
  }
  return Object.fromEntries([...out].map(([chain, letters]) => [chain, letters.join('')]));
}

/**
 * Encode `{chain: 'ACDE...'}` (or a single string for a single-chain spec) into a
 * restype-index array, leaving non-protein tokens at their own identity.
 */
export function encodeSequence(context, sequence) {
  const chains = [...new Set(context.residues.filter((r) => r.kind === 'protein').map((r) => r.chain))];
  let byChain = sequence;
  if (typeof sequence === 'string') {
    if (chains.length !== 1) {
      throw new Error(`spec has chains ${chains.join(', ')}; pass a {chain: sequence} map`);
    }
    byChain = { [chains[0]]: sequence };
  }
  const unknown = Object.keys(byChain).filter((c) => !chains.includes(c));
  if (unknown.length) {
    throw new Error(`chains ${unknown.join(', ')} are not protein chains of this spec`);
  }

  const out = Int32Array.from(context.restypeIndex);
  const cursor = new Map();
  for (let i = 0; i < context.nToken; i += 1) {
    const meta = context.residues[i];
    if (meta.kind !== 'protein') continue;
    const position = cursor.get(meta.chain) ?? 0;
    cursor.set(meta.chain, position + 1);
    const text = byChain[meta.chain];
    if (text === undefined) continue; // chain left out keeps its native identity
    const letter = text[position]?.toUpperCase();
    if (letter === undefined) {
      throw new Error(`sequence for chain ${meta.chain} is shorter than its ${position + 1} residues`);
    }
    const index = ONE_LETTER_TO_INDEX[letter];
    if (index === undefined) throw new Error(`unsupported residue ${letter} in chain ${meta.chain}`);
    out[i] = index;
  }
  for (const [chain, text] of Object.entries(byChain)) {
    const expected = cursor.get(chain) ?? 0;
    if (text.length !== expected) {
      throw new Error(`chain ${chain} has ${expected} residues, sequence has ${text.length}`);
    }
  }
  return out;
}

/**
 * Minimal fixed-column PDB reader producing a spec. Protein residues keep their
 * N/CA/C/O atoms; non-water HETATM groups become context-atom chains.
 *
 * The text is treated purely as data: nothing in it is executed or imported.
 */
export function parsePdbSpec(text, options = {}) {
  const {
    name = 'structure',
    chains: keepChains = null,
    includeHetero = true,
    skipWater = true,
  } = options;
  const waters = new Set(['HOH', 'DOD']);
  const proteinNames = new Set(PROTEIN_THREE);
  const protein = new Map();
  const hetero = new Map();
  const proteinOrder = [];
  const heteroOrder = [];

  for (const line of String(text).split('\n')) {
    const record = line.slice(0, 6).trim();
    if (record !== 'ATOM' && record !== 'HETATM') continue;
    const altloc = line[16];
    if (altloc !== ' ' && altloc !== 'A' && altloc !== undefined) continue;
    const atomName = line.slice(12, 16).trim();
    const resName = line.slice(17, 20).trim();
    const chainId = line[21]?.trim() || 'A';
    if (keepChains && !keepChains.includes(chainId)) continue;
    const resSeq = Number.parseInt(line.slice(22, 26), 10);
    const x = Number.parseFloat(line.slice(30, 38));
    const y = Number.parseFloat(line.slice(38, 46));
    const z = Number.parseFloat(line.slice(46, 54));
    if (!Number.isFinite(resSeq) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }
    const element = (line.slice(76, 78).trim() || atomName[0]).toUpperCase();
    if (element === 'H' || element === 'D') continue;

    if (record === 'ATOM' && proteinNames.has(resName)) {
      const key = `${chainId}|${resSeq}`;
      let residue = protein.get(key);
      if (!residue) {
        residue = { chain: chainId, resName, resSeq };
        protein.set(key, residue);
        proteinOrder.push(residue);
      }
      if (PROTEIN_BACKBONE.includes(atomName) && !residue[atomName]) {
        residue[atomName] = [x, y, z];
      }
    } else if (record === 'HETATM' && includeHetero && !(skipWater && waters.has(resName))) {
      const key = `${chainId}|${resName}|${resSeq}`;
      let group = hetero.get(key);
      if (!group) {
        group = { chain: chainId, resName, resSeq, atoms: [] };
        hetero.set(key, group);
        heteroOrder.push(group);
      }
      group.atoms.push({ name: atomName, element, charge: 0, coord: [x, y, z] });
    }
  }

  const byChain = new Map();
  for (const residue of proteinOrder) {
    if (!residue.CA) continue;
    if (!byChain.has(residue.chain)) byChain.set(residue.chain, []);
    byChain.get(residue.chain).push(residue);
  }
  const specChains = [...byChain].map(([id, residues]) => ({ id, kind: 'protein', residues }));
  for (const group of heteroOrder) {
    specChains.push({ id: group.chain, kind: 'ligand', residues: [group] });
  }
  return { name, chains: specChains };
}
