/**
 * Pure-JS Float32 CPU reference for ArtiosPotts: the 3-layer MPNN encoder,
 * the 3-layer sequence-blind MPNN decoder and the factorized Potts `h`/`J` head.
 *
 * This is the numerical reference the WebGPU implementation is checked against,
 * and the fallback when WebGPU is unavailable. It is a direct transcription of
 * `artios.model.artios_potts.ArtiosPotts` restricted to inference:
 *
 * * `MPNNFeaturizer` — node features from entity flags, context-atom reference
 *   features and non-protein restypes; edge features from 25 backbone-pair
 *   Gaussian RBFs plus the AF3 relative position encoding.
 * * `MPNNEncoderLayer` x 3 — SwiGLU message passing with node and edge updates.
 * * `MPNNDecoderLayer` x 3 — run on an all-mask sequence, so `h` and `J` depend
 *   only on the structure.
 * * `GraphPotts` (`factor` parameterisation) — `h = W_h(node)`,
 *   `J = W_J_left(edge) @ W_J_right(edge)`, then the zero-sum gauge and the
 *   `J_ij(a,b) = (J_ij(a,b) + J_ji(b,a)) / 2` symmetrization.
 *
 * ```js
 * import { createArtiosPottsCPU } from 'artios-webgpu/potts/cpu';
 *
 * const model = await createArtiosPottsCPU({ baseUrl: '/weights/potts', precision: 'fp16' });
 * const context = model.featurize(spec);
 * const params = model.computeParams(context);
 * const score = model.score(params, context, 'MKTAYIAKQRQISFVKSHFSRQ');
 * const design = model.design(params, context, { seed: 0, nSweeps: 150 });
 * ```
 *
 * Dense layers are plain triple loops, so cost scales as `nToken * k * d^2`;
 * a few hundred tokens takes seconds, which is why the WebGPU path exists.
 */

import {
  NUM_ELEMENTS,
  NUM_RESIDUE_TYPES,
  backboneDistanceRbf,
  buildNeighbourGraph,
  encodeSequence,
  featurize,
  loadWeightBundle,
  relativePositionFeatures,
  sampleFromParams,
  scoreSequence,
  sequencesByChain,
  withoutLigand,
} from '../shared/index.js';
import { concatLast, gatherNodes, layerNorm, linear, maskRows, swiGluMlp } from './ops.js';

/** Collect the tensors of one encoder layer. */
function encoderLayer(weights, i) {
  const at = (name) => weights.get(`mpnn.encoder.${i}.${name}`).data;
  return {
    norm1: { weight: at('norm1.weight'), bias: at('norm1.bias') },
    nodeMessage: { wIn: at('node_message_mlp.W_in.weight'), wOut: at('node_message_mlp.W_out.weight') },
    dense: { wIn: at('dense.W_in.weight'), wOut: at('dense.W_out.weight') },
    norm2: { weight: at('norm2.weight'), bias: at('norm2.bias') },
    edgeUpdate: { wIn: at('edge_update_mlp.W_in.weight'), wOut: at('edge_update_mlp.W_out.weight') },
    norm3: { weight: at('norm3.weight'), bias: at('norm3.bias') },
  };
}

/** Collect the tensors of one decoder layer. */
function decoderLayer(weights, i) {
  const at = (name) => weights.get(`mpnn.decoder.${i}.${name}`).data;
  return {
    nodeMessage: { wIn: at('node_message_mlp.W_in.weight'), wOut: at('node_message_mlp.W_out.weight') },
    norm1: { weight: at('norm1.weight'), bias: at('norm1.bias') },
    dense: { wIn: at('dense.W_in.weight'), wOut: at('dense.W_out.weight') },
    norm2: { weight: at('norm2.weight'), bias: at('norm2.bias') },
  };
}

function collectWeights(weights, config) {
  const at = (name) => weights.get(name).data;
  return {
    featurizer: {
      entityTypeEmbed: at('mpnn.featurizer.entity_type_embed.weight'),
      nonProteinRestypeEmbed: at('mpnn.featurizer.non_protein_restype_embed.weight'),
      ligandConditionEmbed: at('mpnn.featurizer.ligand_condition_embed.weight'),
      relposEmbed: at('mpnn.featurizer.positional_encoder.relpos.embed_relpos_features.weight'),
      rbfCenters: at('mpnn.featurizer.rbf.centers'),
      rbfGamma: at('mpnn.featurizer.rbf.gamma')[0],
      embedEdge: at('mpnn.featurizer.embed_edge.weight'),
      edgeNorm: { weight: at('mpnn.featurizer.edge_norm.weight'), bias: at('mpnn.featurizer.edge_norm.bias') },
      nodeNorm: { weight: at('mpnn.featurizer.node_norm.weight'), bias: at('mpnn.featurizer.node_norm.bias') },
    },
    encoder: Array.from({ length: config.num_encoder_layers }, (_, i) => encoderLayer(weights, i)),
    decoder: Array.from({ length: config.num_decoder_layers }, (_, i) => decoderLayer(weights, i)),
    embedRestype: at('mpnn.embed_restype.weight'),
    linearOut: at('mpnn.linear_out.weight'),
    potts: {
      logScale: at('potts.log_scale')[0],
      wH: at('potts.W_h.weight'),
      bH: at('potts.W_h.bias'),
      wJLeft: at('potts.W_J_left.weight'),
      bJLeft: at('potts.W_J_left.bias'),
      wJRight: at('potts.W_J_right.weight'),
      bJRight: at('potts.W_J_right.bias'),
    },
  };
}

/**
 * `MPNNFeaturizer._compute_node_features`: context-atom reference features
 * averaged to tokens, entity-flag embeddings and non-protein restype embeddings,
 * summed and layer-normed.
 */
function nodeFeatures(context, w, config) {
  const { nToken, nAtom, numResidueTypes: R } = { ...context, numResidueTypes: NUM_RESIDUE_TYPES };
  const nodeDim = config.node_dim;
  const eps = config.layer_norm_eps ?? 1e-5;

  // ligand_input = [ref_charge, ref_mask, ref_element_one_hot]  -> [nAtom, 130]
  const ligandInput = new Float32Array(nAtom * (2 + NUM_ELEMENTS));
  for (let a = 0; a < nAtom; a += 1) {
    const base = a * (2 + NUM_ELEMENTS);
    ligandInput[base] = context.refCharge[a];
    ligandInput[base + 1] = context.refMask[a];
    ligandInput[base + 2 + context.refElementIndex[a]] = 1;
  }
  const ligandAtom = linear(
    ligandInput, nAtom, 2 + NUM_ELEMENTS, w.featurizer.ligandConditionEmbed, nodeDim,
  );
  // is_atomic = is_ligand | is_motif_related, at atom level
  for (let a = 0; a < nAtom; a += 1) {
    if (context.isLigandAtom[a] || context.isMotifRelatedAtom[a]) continue;
    ligandAtom.fill(0, a * nodeDim, a * nodeDim + nodeDim);
  }

  // average_atom_to_token: sum / (count + 1e-4)
  const ligandToken = new Float32Array(nToken * nodeDim);
  const counts = new Float32Array(nToken);
  for (let a = 0; a < nAtom; a += 1) {
    if (!context.atomMask[a]) continue;
    const token = context.atomToTokenIdx[a];
    counts[token] += 1;
    for (let c = 0; c < nodeDim; c += 1) {
      ligandToken[token * nodeDim + c] += ligandAtom[a * nodeDim + c];
    }
  }
  for (let i = 0; i < nToken; i += 1) {
    const denominator = (counts[i] > 0 ? 1 : 0) + 1e-4;
    for (let c = 0; c < nodeDim; c += 1) ligandToken[i * nodeDim + c] /= denominator;
  }

  // entity flags, in the order MPNNFeaturizer.entity_features declares them
  const entityInput = new Float32Array(nToken * 4);
  for (let i = 0; i < nToken; i += 1) {
    entityInput[i * 4] = context.isProteinToken[i];
    entityInput[i * 4 + 1] = context.isLigandToken[i];
    entityInput[i * 4 + 2] = context.isDnaToken[i];
    entityInput[i * 4 + 3] = context.isRnaToken[i];
  }
  const entityFeats = linear(entityInput, nToken, 4, w.featurizer.entityTypeEmbed, nodeDim);

  // restype * (1 - is_protein_token)
  const restypeInput = new Float32Array(nToken * R);
  for (let i = 0; i < nToken; i += 1) {
    if (context.isProteinToken[i]) continue;
    for (let a = 0; a < R; a += 1) restypeInput[i * R + a] = context.restype[i * R + a];
  }
  const restypeFeats = linear(
    restypeInput, nToken, R, w.featurizer.nonProteinRestypeEmbed, nodeDim,
  );

  const summed = new Float32Array(nToken * nodeDim);
  for (let i = 0; i < summed.length; i += 1) {
    summed[i] = ligandToken[i] + entityFeats[i] + restypeFeats[i];
  }
  return layerNorm(
    summed, nToken, nodeDim, w.featurizer.nodeNorm.weight, w.featurizer.nodeNorm.bias, eps,
  );
}

/** `MPNNFeaturizer.forward`: node and edge embeddings on the kNN graph. */
function runFeaturizer(context, graph, w, config) {
  const { nToken } = context;
  const { k, edgeIndex, edgeMask } = graph;
  const nodeDim = config.node_dim;
  const edgeDim = config.edge_dim;
  const eps = config.layer_norm_eps ?? 1e-5;

  const rbf = backboneDistanceRbf(context, graph, {
    numRbf: config.num_rbf_kernels,
    minDistance: config.rbf_min_distance,
    maxDistance: config.rbf_max_distance,
    // The checkpoint stores centers/gamma as buffers; use them verbatim.
    centers: w.featurizer.rbfCenters,
    gamma: w.featurizer.rbfGamma,
  });
  const relpos = relativePositionFeatures(context, graph, {
    rMax: config.relpos_r_max,
    sMax: config.relpos_s_max,
  });

  // embed_edge takes [25 * numRbf + positionalEncodingDim]; the relative position
  // features are embedded first, exactly as EdgeRelativePositionalEncoding does.
  const relposEmbedded = linear(
    relpos.data, nToken * k, relpos.featureDim, w.featurizer.relposEmbed,
    config.positional_encoding_dim,
  );
  const edgeInput = concatLast(
    [[rbf.data, rbf.featureDim], [relposEmbedded, config.positional_encoding_dim]],
    nToken * k,
  );
  let edgeFeats = linear(
    edgeInput, nToken * k, rbf.featureDim + config.positional_encoding_dim,
    w.featurizer.embedEdge, edgeDim,
  );
  edgeFeats = layerNorm(
    edgeFeats, nToken * k, edgeDim,
    w.featurizer.edgeNorm.weight, w.featurizer.edgeNorm.bias, eps,
  );

  let nodeFeats = nodeFeatures(context, w, config);
  maskRows(nodeFeats, nToken, nodeDim, context.tokenMask);
  // edge_embedding *= gather_nodes(token_mask, edge_index) — the neighbour's mask.
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      if (context.tokenMask[edgeIndex[i * k + s]]) continue;
      const base = (i * k + s) * edgeDim;
      edgeFeats.fill(0, base, base + edgeDim);
    }
  }
  return { nodeFeats, edgeFeats, edgeIndex, edgeMask, k };
}

/** `BaseMPNNLayer.gather_and_concat_neighbourhood`: `[self, edge, neighbour]`. */
function neighbourhood(nodeFeats, edgeFeats, edgeIndex, nToken, k, nodeDim, edgeDim) {
  const selfExpanded = new Float32Array(nToken * k * nodeDim);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const dst = (i * k + s) * nodeDim;
      for (let c = 0; c < nodeDim; c += 1) selfExpanded[dst + c] = nodeFeats[i * nodeDim + c];
    }
  }
  const neighbours = gatherNodes(nodeFeats, nodeDim, edgeIndex, nToken, k);
  return concatLast(
    [[selfExpanded, nodeDim], [edgeFeats, edgeDim], [neighbours, nodeDim]],
    nToken * k,
  );
}

/** `MPNNEncoderLayer.forward`. */
function runEncoderLayer(layer, state, context, config) {
  const { nToken } = context;
  const { k, edgeIndex, edgeMask } = state;
  const nodeDim = config.node_dim;
  const edgeDim = config.edge_dim;
  const scale = config.num_neighbours;
  const eps = config.layer_norm_eps ?? 1e-5;
  const rows = nToken * k;
  let { nodeFeats, edgeFeats } = state;

  // --- node message ---
  let hNn = neighbourhood(nodeFeats, edgeFeats, edgeIndex, nToken, k, nodeDim, edgeDim);
  let message = swiGluMlp(hNn, rows, 2 * nodeDim + edgeDim, layer.nodeMessage.wIn, layer.nodeMessage.wOut, nodeDim);
  maskRows(message, rows, nodeDim, edgeMask);
  const dh = new Float32Array(nToken * nodeDim);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const src = (i * k + s) * nodeDim;
      for (let c = 0; c < nodeDim; c += 1) dh[i * nodeDim + c] += message[src + c];
    }
    for (let c = 0; c < nodeDim; c += 1) dh[i * nodeDim + c] /= scale;
  }
  const residual = new Float32Array(nToken * nodeDim);
  for (let i = 0; i < residual.length; i += 1) residual[i] = nodeFeats[i] + dh[i];
  nodeFeats = layerNorm(residual, nToken, nodeDim, layer.norm1.weight, layer.norm1.bias, eps);

  // --- node feedforward ---
  const dense = swiGluMlp(nodeFeats, nToken, nodeDim, layer.dense.wIn, layer.dense.wOut, nodeDim);
  for (let i = 0; i < dense.length; i += 1) dense[i] += nodeFeats[i];
  nodeFeats = layerNorm(dense, nToken, nodeDim, layer.norm2.weight, layer.norm2.bias, eps);
  maskRows(nodeFeats, nToken, nodeDim, context.tokenMask);

  // --- edge update, on the *updated* node features ---
  hNn = neighbourhood(nodeFeats, edgeFeats, edgeIndex, nToken, k, nodeDim, edgeDim);
  message = swiGluMlp(hNn, rows, 2 * nodeDim + edgeDim, layer.edgeUpdate.wIn, layer.edgeUpdate.wOut, edgeDim);
  for (let i = 0; i < message.length; i += 1) message[i] += edgeFeats[i];
  edgeFeats = layerNorm(message, rows, edgeDim, layer.norm3.weight, layer.norm3.bias, eps);
  maskRows(edgeFeats, rows, edgeDim, edgeMask);

  return { ...state, nodeFeats, edgeFeats };
}

/** `MPNN.encode`. */
function runEncoder(context, graph, w, config) {
  let state = runFeaturizer(context, graph, w, config);
  const featurizerOutput = {
    nodeFeats: Float32Array.from(state.nodeFeats),
    edgeFeats: Float32Array.from(state.edgeFeats),
  };
  for (const layer of w.encoder) state = runEncoderLayer(layer, state, context, config);
  return { ...state, featurizerOutput };
}

/**
 * `MPNN.decode` with the all-mask sequence input, so `h` and `J` are
 * sequence-blind (`ArtiosPotts._build_sequence_blind_inputs`).
 *
 * Returns the decoder node features *and* the augmented edge features from the
 * last loop iteration — note those are built from the node features as they were
 * entering the final decoder layer, which is what the reference feeds to the
 * Potts head.
 */
function runDecoder(context, encoded, w, config) {
  const { nToken } = context;
  const { k, edgeIndex, edgeMask } = encoded;
  const nodeDim = config.node_dim;
  const edgeDim = config.edge_dim;
  const scale = config.num_neighbours;
  const eps = config.layer_norm_eps ?? 1e-5;
  const rows = nToken * k;
  const vocab = NUM_RESIDUE_TYPES + 1;

  // pad_and_mask_restypes: every valid token becomes the mask token.
  const maskedRestypes = new Float32Array(nToken * vocab);
  for (let i = 0; i < nToken; i += 1) {
    if (context.tokenMask[i]) maskedRestypes[i * vocab + config.mask_token_index] = 1;
    // Invalid tokens keep the all-zero padded restype (restypes are blanked out).
  }
  const sequenceEmbeds = linear(maskedRestypes, nToken, vocab, w.embedRestype, nodeDim);
  const neighbourSequence = gatherNodes(sequenceEmbeds, nodeDim, edgeIndex, nToken, k);
  const sequenceEdgeFeats = concatLast(
    [[encoded.edgeFeats, edgeDim], [neighbourSequence, nodeDim]], rows,
  );
  const sequenceEdgeDim = edgeDim + nodeDim;
  const decoderEdgeDim = edgeDim + 2 * nodeDim;

  let nodeFeats = encoded.nodeFeats;
  let decoderEdgeFeats = null;
  for (const layer of w.decoder) {
    const neighbourNodes = gatherNodes(nodeFeats, nodeDim, edgeIndex, nToken, k);
    decoderEdgeFeats = concatLast(
      [[sequenceEdgeFeats, sequenceEdgeDim], [neighbourNodes, nodeDim]], rows,
    );

    const selfExpanded = new Float32Array(rows * nodeDim);
    for (let i = 0; i < nToken; i += 1) {
      for (let s = 0; s < k; s += 1) {
        const dst = (i * k + s) * nodeDim;
        for (let c = 0; c < nodeDim; c += 1) selfExpanded[dst + c] = nodeFeats[i * nodeDim + c];
      }
    }
    const hEV = concatLast([[selfExpanded, nodeDim], [decoderEdgeFeats, decoderEdgeDim]], rows);
    const message = swiGluMlp(
      hEV, rows, nodeDim + decoderEdgeDim, layer.nodeMessage.wIn, layer.nodeMessage.wOut, nodeDim,
    );
    maskRows(message, rows, nodeDim, edgeMask);

    const dh = new Float32Array(nToken * nodeDim);
    for (let i = 0; i < nToken; i += 1) {
      for (let s = 0; s < k; s += 1) {
        const src = (i * k + s) * nodeDim;
        for (let c = 0; c < nodeDim; c += 1) dh[i * nodeDim + c] += message[src + c];
      }
      for (let c = 0; c < nodeDim; c += 1) dh[i * nodeDim + c] /= scale;
    }
    const residual = new Float32Array(nToken * nodeDim);
    for (let i = 0; i < residual.length; i += 1) residual[i] = nodeFeats[i] + dh[i];
    nodeFeats = layerNorm(residual, nToken, nodeDim, layer.norm1.weight, layer.norm1.bias, eps);

    const dense = swiGluMlp(nodeFeats, nToken, nodeDim, layer.dense.wIn, layer.dense.wOut, nodeDim);
    for (let i = 0; i < dense.length; i += 1) dense[i] += nodeFeats[i];
    nodeFeats = layerNorm(dense, nToken, nodeDim, layer.norm2.weight, layer.norm2.bias, eps);
    maskRows(nodeFeats, nToken, nodeDim, context.tokenMask);
  }

  const logits = linear(nodeFeats, nToken, nodeDim, w.linearOut, NUM_RESIDUE_TYPES);
  return { nodeFeats, decoderEdgeFeats, decoderEdgeDim, logits, maskedRestypes };
}

/** `_zero_sum_gauge` on `h` `[nToken, R]`, in place. */
function zeroSumGaugeH(h, nToken, R) {
  for (let i = 0; i < nToken; i += 1) {
    let mean = 0;
    for (let a = 0; a < R; a += 1) mean += h[i * R + a];
    mean /= R;
    for (let a = 0; a < R; a += 1) h[i * R + a] -= mean;
  }
  return h;
}

/**
 * `_zero_sum_gauge` on one `[R, R]` block, in place:
 * `J - mean_b - mean_a + mean_ab`.
 */
function zeroSumGaugeBlock(J, base, R) {
  const rowMean = new Float64Array(R);
  const colMean = new Float64Array(R);
  let grand = 0;
  for (let a = 0; a < R; a += 1) {
    let sum = 0;
    for (let b = 0; b < R; b += 1) {
      const v = J[base + a * R + b];
      sum += v;
      colMean[b] += v;
    }
    rowMean[a] = sum / R;
    grand += sum;
  }
  grand /= R * R;
  for (let b = 0; b < R; b += 1) colMean[b] /= R;
  for (let a = 0; a < R; a += 1) {
    for (let b = 0; b < R; b += 1) {
      J[base + a * R + b] += grand - rowMean[a] - colMean[b];
    }
  }
}

/**
 * `_symmetrize_J`: `J_ij(a, b) <- (J_ij(a, b) + J_ji(b, a)) / 2`, zeroing edges
 * whose reverse is missing from the directed kNN graph.
 *
 * Done in place by visiting each `(i, k) <-> (j, k')` pair once, which halves the
 * peak memory of the largest tensor in the model.
 */
function symmetrizeJ(J, edgeIndex, edgeMask, nToken, k, R) {
  const block = R * R;
  const reverseSlot = new Int32Array(nToken * k).fill(-1);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const j = edgeIndex[i * k + s];
      for (let t = 0; t < k; t += 1) {
        if (edgeIndex[j * k + t] === i) { reverseSlot[i * k + s] = t; break; }
      }
    }
  }

  const processed = new Uint8Array(nToken * k);
  const a = new Float32Array(block);
  const b = new Float32Array(block);
  for (let i = 0; i < nToken; i += 1) {
    for (let s = 0; s < k; s += 1) {
      const edge = i * k + s;
      if (processed[edge]) continue;
      const reverse = reverseSlot[edge];
      if (reverse < 0) { // has_reverse == 0
        J.fill(0, edge * block, edge * block + block);
        processed[edge] = 1;
        continue;
      }
      const other = edgeIndex[edge] * k + reverse;
      a.set(J.subarray(edge * block, edge * block + block));
      b.set(J.subarray(other * block, other * block + block));
      const scale = edgeMask[edge] ? 0.5 : 0;
      const otherScale = edgeMask[other] ? 0.5 : 0;
      for (let p = 0; p < R; p += 1) {
        for (let q = 0; q < R; q += 1) {
          J[edge * block + p * R + q] = scale * (a[p * R + q] + b[q * R + p]);
        }
      }
      if (other !== edge) {
        for (let p = 0; p < R; p += 1) {
          for (let q = 0; q < R; q += 1) {
            J[other * block + p * R + q] = otherScale * (b[p * R + q] + a[q * R + p]);
          }
        }
        processed[other] = 1;
      }
      processed[edge] = 1;
    }
  }
  return J;
}

/** `GraphPotts.forward`, `factor` parameterisation. */
function runPottsHead(context, encoded, decoded, w, config) {
  const { nToken } = context;
  const { k, edgeIndex, edgeMask } = encoded;
  const R = config.num_residue_types;
  const F = config.potts.num_factors;
  const rows = nToken * k;
  const scale = Math.exp(w.potts.logScale);

  if (config.potts.parameterisation !== 'factor') {
    throw new Error(
      `this port implements the "factor" Potts parameterisation; the bundle says ` +
      `"${config.potts.parameterisation}"`,
    );
  }

  const h = linear(decoded.nodeFeats, nToken, config.node_dim, w.potts.wH, R, w.potts.bH);
  for (let i = 0; i < nToken; i += 1) {
    const maskH = scale * (context.tokenMask[i] ? 1 : 0);
    for (let a = 0; a < R; a += 1) h[i * R + a] *= maskH;
  }

  const left = linear(
    decoded.decoderEdgeFeats, rows, decoded.decoderEdgeDim, w.potts.wJLeft, R * F, w.potts.bJLeft,
  );
  const right = linear(
    decoded.decoderEdgeFeats, rows, decoded.decoderEdgeDim, w.potts.wJRight, F * R, w.potts.bJRight,
  );
  // mask_J multiplies *both* factors, so J picks up scale^2 — as in the reference.
  for (let e = 0; e < rows; e += 1) {
    const maskJ = scale * (edgeMask[e] ? 1 : 0);
    for (let c = 0; c < R * F; c += 1) {
      left[e * R * F + c] *= maskJ;
      right[e * F * R + c] *= maskJ;
    }
  }

  const block = R * R;
  const J = new Float32Array(rows * block);
  for (let e = 0; e < rows; e += 1) {
    const lBase = e * R * F;
    const rBase = e * F * R;
    const jBase = e * block;
    for (let p = 0; p < R; p += 1) {
      for (let q = 0; q < R; q += 1) {
        let acc = 0;
        for (let f = 0; f < F; f += 1) acc += left[lBase + p * F + f] * right[rBase + f * R + q];
        J[jBase + p * R + q] = acc;
      }
    }
  }

  if (config.potts.zero_sum_gauge) {
    zeroSumGaugeH(h, nToken, R);
    for (let e = 0; e < rows; e += 1) zeroSumGaugeBlock(J, e * block, R);
  }
  if (config.potts.symmetric_J) {
    symmetrizeJ(J, edgeIndex, edgeMask, nToken, k, R);
  }

  return { h, J };
}

/** Box-Muller gaussian noise from a uniform generator. */
function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/**
 * Load the released weights and build a CPU ArtiosPotts model.
 *
 * @param {object} options
 * @param {string} options.baseUrl directory holding the exporter's `manifest.json`
 * @param {'fp32'|'fp16'|'int8'} [options.precision]
 * @param {(url: string) => Promise<Uint8Array|ArrayBuffer>} [options.read]
 *   how to fetch a bundle file; defaults to `fetch`
 * @param {boolean} [options.verify] check the manifest SHA-256 of every file
 */
export async function createArtiosPottsCPU({ baseUrl, precision = 'fp32', read, verify = false } = {}) {
  const weights = await loadWeightBundle({ baseUrl, precision, read, verify });
  const config = weights.config;
  if (!config) throw new Error(`bundle at ${baseUrl} has no config in its manifest`);
  const w = collectWeights(weights, config);

  /** Run the encoder, decoder and Potts head once on one context. */
  const forward = (context, { trace = false } = {}) => {
    const graph = buildNeighbourGraph(context, config.num_neighbours);
    const encoded = runEncoder(context, graph, w, config);
    const decoded = runDecoder(context, encoded, w, config);
    const { h, J } = runPottsHead(context, encoded, decoded, w, config);
    const params = {
      h,
      J,
      edgeIndex: encoded.edgeIndex,
      edgeMask: encoded.edgeMask,
      nToken: context.nToken,
      k: encoded.k,
      numResidueTypes: config.num_residue_types,
    };
    if (!trace) return params;
    return {
      ...params,
      trace: {
        featNode: encoded.featurizerOutput.nodeFeats,
        featEdge: encoded.featurizerOutput.edgeFeats,
        encNode: encoded.nodeFeats,
        encEdge: encoded.edgeFeats,
        decNode: decoded.nodeFeats,
        decEdge: decoded.decoderEdgeFeats,
        decEdgeDim: decoded.decoderEdgeDim,
        mpnnLogits: decoded.logits,
        maskedRestypes: decoded.maskedRestypes,
      },
    };
  };

  return {
    /** The manifest config the weights were exported with. */
    config,
    /** The loaded weight bundle (`precision`, `names`, `get`, ...). */
    weights,
    precision,
    /** `featurize`, re-exported so callers need only this module. */
    featurize,
    encodeSequence: (context, sequence) => encodeSequence(context, sequence),
    sequencesByChain: (context, sequence) => sequencesByChain(context, sequence),

    /**
     * Ensemble-averaged Potts parameters for one structure
     * — `ArtiosPotts.compute_potts_params`.
     *
     * Defaults to a single noise-free pass, which is what the oracle fixtures
     * record. The released inference config instead uses `ensembleSize: 50` with
     * `ensembleNoiseScale: 0.25`, which perturbs only the backbone coordinates
     * (so the kNN graph is shared across members) and averages `h` and `J`.
     *
     * @param {object|ReturnType<typeof featurize>} input a spec or a context
     * @param {object} [options]
     * @param {number} [options.ensembleSize]
     * @param {number} [options.ensembleNoiseScale]
     * @param {boolean} [options.unconditional] also compute the ligand-free
     *   `h`/`J` used for classifier-free guidance
     * @param {boolean} [options.trace] keep every intermediate activation
     */
    computeParams(input, options = {}) {
      const {
        ensembleSize = 1,
        ensembleNoiseScale = 0,
        unconditional = false,
        trace = false,
        seed = 0,
        rng = null,
      } = options;
      const context = input.nToken ? input : featurize(input);

      const runEnsemble = (ctx) => {
        if (ensembleSize <= 1 || ensembleNoiseScale <= 0) return forward(ctx, { trace });
        const draw = rng ?? ((() => {
          // Local import avoids a cycle at module scope.
          let state = seed >>> 0;
          return () => {
            state = (state * 1664525 + 1013904223) >>> 0;
            return state / 4294967296;
          };
        })());
        let accumulated = null;
        for (let member = 0; member < ensembleSize; member += 1) {
          const noisy = Float32Array.from(ctx.backboneCoordinate);
          for (let i = 0; i < noisy.length; i += 1) {
            noisy[i] += ensembleNoiseScale * gaussian(draw);
          }
          const params = forward({ ...ctx, backboneCoordinate: noisy }, { trace: false });
          if (!accumulated) {
            accumulated = { ...params, h: Float32Array.from(params.h), J: Float32Array.from(params.J) };
          } else {
            for (let i = 0; i < accumulated.h.length; i += 1) accumulated.h[i] += params.h[i];
            for (let i = 0; i < accumulated.J.length; i += 1) accumulated.J[i] += params.J[i];
          }
        }
        for (let i = 0; i < accumulated.h.length; i += 1) accumulated.h[i] /= ensembleSize;
        for (let i = 0; i < accumulated.J.length; i += 1) accumulated.J[i] /= ensembleSize;
        return accumulated;
      };

      const params = runEnsemble(context);
      params.context = context;
      if (unconditional) params.uncond = runEnsemble(withoutLigand(context));
      return params;
    },

    /**
     * Potts energy and pseudo-log-likelihood of a sequence — `PottsDesigner.score`.
     *
     * `sequence` may be a restype-index array, a single string (single-chain
     * specs) or a `{chain: sequence}` map.
     */
    score(params, context, sequence = null) {
      const indices = sequence === null
        ? context.restypeIndex
        : (typeof sequence === 'object' && sequence.length !== undefined
          ? sequence
          : encodeSequence(context, sequence));
      return {
        ...scoreSequence(params, indices, { isProteinToken: context.isProteinToken }),
        sequence: indices,
        sequences: sequencesByChain(context, indices),
      };
    },

    /**
     * Sample a sequence from `params` — `ArtiosPotts.sample_from_params`.
     *
     * Options are the sampling-config knobs: `temperature`, `temperatureInit`,
     * `annealingFraction`, `nSweeps`, `proposal`, `fixedMask`, `excludeCys`,
     * `excludeMet`, `energyBias`, `groupEnergyBias`, `cfgScale` with `uncond`,
     * and the penalty weights.
     */
    design(params, context, options = {}) {
      const result = sampleFromParams(params, {
        context,
        uncond: params.uncond ?? options.uncond ?? null,
        ...options,
      });
      return { ...result, sequences: sequencesByChain(context, result.sequence) };
    },

    /** Featurize, compute parameters and sample, in one call. */
    run(spec, options = {}) {
      const context = featurize(spec);
      const params = this.computeParams(context, options);
      const design = this.design(params, context, options);
      return { context, params, design };
    },

    /** Escape hatch for tests and the WebGPU parity harness. */
    forward,
  };
}

export { featurize, scoreSequence, sampleFromParams };
export * as ops from './ops.js';
