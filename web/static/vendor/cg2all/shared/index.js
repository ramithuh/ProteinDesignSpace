// cg2all (huhlim/cg2all, Apache-2.0) Cα → all-atom backmapping — backend-independent pipeline.
//   prepareInput(pdb)  → Cα graph, node/edge features, SE(3) basis      (this file + features.js)
//   engine.predict(g)  → structure-module output s (N×20), v (N×3×3)     (cpu/ or webgpu/)
//   assemble(g, out)   → rigid-group all-atom model, termini, PDB text   (build.js)

import { buildAtoms, decodeOutputs, residueAtoms, toPdb } from './build.js';
import {
  computeBasis,
  computeContinuity,
  edgeFeatures,
  incomingCsr,
  nodeFeatures,
  radiusGraph,
} from './features.js';
import { parseCaTrace, residueTypeIndex, threadSequence } from './structure.js';

export { buildNetwork, forwardCpu } from './model.js';
export { loadWeights, decodeTensors } from './weights.js';
export { parseCaTrace, threadSequence, splitSequence } from './structure.js';
export * as features from './features.js';
export * as build from './build.js';

export const VARIANTS = {
  ca: { checkpoint: 'CalphaBasedModel', fixAtom: false },
  'ca-fix': { checkpoint: 'CalphaBasedModel-FIX', fixAtom: true },
};

/** Build the model input for a PDB (Å) — optionally threading a new sequence onto the trace first. */
export function prepareInput(pdbText, { sequence } = {}) {
  const trace = parseCaTrace(pdbText);
  if (sequence !== undefined && sequence !== null && sequence !== '') threadSequence(trace, sequence);
  return prepareFromTrace(trace);
}

/** Build the model input from a parsed trace (`parseCaTrace` shape; Cα in Å). */
export function prepareFromTrace(trace) {
  const n = trace.residues.length;
  if (n === 0) throw new Error('cg2all: the input structure has no residues with a CA atom');
  const pos = new Float64Array(n * 3);
  const residueType = new Int32Array(n);
  const chainIndex = new Int32Array(n);
  trace.residues.forEach((r, i) => {
    pos[3 * i] = r.ca[0] / 10;
    pos[3 * i + 1] = r.ca[1] / 10;
    pos[3 * i + 2] = r.ca[2] / 10;
    residueType[i] = residueTypeIndex(r.resName);
    chainIndex[i] = r.chainIndex;
  });
  const continuity = computeContinuity(pos, chainIndex);
  const { f0, f1 } = nodeFeatures(pos, continuity);
  const graph = radiusGraph(pos);
  const { feat: edgeFeat, relPos } = edgeFeatures(pos, graph, continuity, trace.ssbonds);
  const basis = computeBasis(relPos);
  return {
    trace,
    n,
    e: graph.src.length,
    pos,
    residueType,
    chainIndex,
    continuity,
    f0,
    f1,
    src: graph.src,
    dst: graph.dst,
    csr: incomingCsr(graph, n),
    edgeFeat,
    relPos,
    basis,
  };
}

/** Count of chain breaks inside chains (consecutive same-chain residues with Cα–Cα > 10 Å). */
export function countChainBreaks(prepared) {
  let breaks = 0;
  for (let i = 1; i < prepared.n; i++) {
    if (prepared.chainIndex[i] === prepared.chainIndex[i - 1] && !prepared.continuity.prev[i]) breaks++;
  }
  return breaks;
}

/**
 * Turn structure-module outputs into an all-atom model.
 * @returns {{ pdbText: string, atomCount: number, R: Float64Array, ss: Int32Array, atoms: any[], warnings: string[] }}
 */
export function assemble(prepared, output, { fixAtom = false, hydrogens = false, ligands = true, remarks } = {}) {
  const decoded = decodeOutputs(output.s, output.v, prepared.n, { fixAtom });
  const R = buildAtoms(decoded, prepared.residueType, prepared.pos);
  const atoms = residueAtoms(R, prepared.residueType, prepared.trace.residues, prepared.trace.ssbonds);
  const warnings = [...prepared.trace.warnings];
  const unknown = prepared.trace.residues.filter((_, i) => !atoms[i]).map((r) => `${r.chainId}${r.resNum}${r.insCode}:${r.inputName}`);
  if (unknown.length) {
    warnings.push(`cg2all has no template for ${unknown.length} residue(s) (${unknown.slice(0, 5).join(', ')}${unknown.length > 5 ? ', …' : ''}); their input atoms were kept`);
  }
  for (let i = 0; i < R.length; i++) {
    if (!Number.isFinite(R[i])) throw new Error('cg2all produced non-finite coordinates');
  }
  const { pdbText, atomCount } = toPdb(prepared.trace, atoms, { hydrogens, ligands, remarks });
  return { pdbText, atomCount, R, ss: decoded.ss, atoms, warnings };
}

/** One-call conversion with any engine exposing `predict(prepared)`. */
export async function convert(engine, pdbText, opts = {}) {
  const prepared = prepareInput(pdbText, { sequence: opts.sequence });
  const output = await engine.predict(prepared);
  return { prepared, ...assemble(prepared, output, opts) };
}
