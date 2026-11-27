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
import { filterCysSsbonds, parseCaTrace, residueTypeIndex, threadSequence } from './structure.js';

export { buildNetwork, forwardCpu } from './model.js';
export { loadWeights, decodeTensors } from './weights.js';
export { filterCysSsbonds, parseCaTrace, threadSequence, splitSequence } from './structure.js';
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
  const warnings = [...(trace.warnings ?? [])];
  const rawSsbonds = trace.ssbonds ?? [];
  const ssbonds = filterCysSsbonds(trace.residues, rawSsbonds);
  if (ssbonds.length !== rawSsbonds.length) {
    warnings.push(`cg2all: ignored ${rawSsbonds.length - ssbonds.length} non-CYS SSBOND pair(s).`);
  }
  const continuity = computeContinuity(pos, chainIndex);
  const { f0, f1 } = nodeFeatures(pos, continuity);
  const graph = radiusGraph(pos);
  const { feat: edgeFeat, relPos } = edgeFeatures(pos, graph, continuity, ssbonds);
  const basis = computeBasis(relPos);

  // Warn on L === 2 or collinear chains (unconstrained rotation around the chain axis) and on
  // intra-chain gaps in (4.5, 10.0] Å bridged as bonded by upstream's 1.0 nm cutoff.
  const chains = new Map();
  for (let i = 0; i < n; i++) {
    const c = chainIndex[i];
    if (!chains.has(c)) chains.set(c, []);
    chains.get(c).push(i);
  }
  for (const [c, indices] of chains) {
    const chainLabel = trace.residues[indices[0]].chainId || String(c + 1);
    if (indices.length === 2) {
      warnings.push(
        `cg2all: chain ${chainLabel} has only 2 Cα atoms (L === 2); rotation around the Cα–Cα axis is unconstrained.`,
      );
    } else if (indices.length >= 3) {
      const p0 = trace.residues[indices[0]].ca;
      let maxDist = 0;
      let pMax = p0;
      for (let k = 1; k < indices.length; k++) {
        const pk = trace.residues[indices[k]].ca;
        const d = Math.hypot(pk[0] - p0[0], pk[1] - p0[1], pk[2] - p0[2]);
        if (d > maxDist) {
          maxDist = d;
          pMax = pk;
        }
      }
      if (maxDist <= 1e-6) {
        warnings.push(
          `cg2all: chain ${chainLabel} has collinear/coincident Cα atoms; rotation around the chain axis is unconstrained.`,
        );
      } else {
        const ux = (pMax[0] - p0[0]) / maxDist;
        const uy = (pMax[1] - p0[1]) / maxDist;
        const uz = (pMax[2] - p0[2]) / maxDist;
        let maxPerp = 0;
        for (const idx of indices) {
          const pk = trace.residues[idx].ca;
          const dx = pk[0] - p0[0];
          const dy = pk[1] - p0[1];
          const dz = pk[2] - p0[2];
          const proj = dx * ux + dy * uy + dz * uz;
          const perp = Math.hypot(dx - proj * ux, dy - proj * uy, dz - proj * uz);
          if (perp > maxPerp) maxPerp = perp;
        }
        if (maxPerp < 1e-2) {
          warnings.push(
            `cg2all: chain ${chainLabel} has collinear Cα atoms; rotation around the chain axis is unconstrained.`,
          );
        }
      }
    }
  }
  const bridgedGaps = [];
  for (let i = 1; i < n; i++) {
    if (chainIndex[i] === chainIndex[i - 1] && continuity.prev[i]) {
      const a = trace.residues[i - 1];
      const b = trace.residues[i];
      const dAng = Math.hypot(b.ca[0] - a.ca[0], b.ca[1] - a.ca[1], b.ca[2] - a.ca[2]);
      if (dAng > 4.5) {
        bridgedGaps.push(`${a.chainId || '_'}${a.resNum}${a.insCode}-${b.chainId || '_'}${b.resNum}${b.insCode} (${dAng.toFixed(1)} Å)`);
      }
    }
  }
  if (bridgedGaps.length) {
    warnings.push(
      `cg2all: ${bridgedGaps.length} intra-chain Cα–Cα gap(s) in (4.5, 10.0] Å (${bridgedGaps.slice(0, 3).join(', ')}${bridgedGaps.length > 3 ? ', …' : ''}) treated as continuous by upstream's 10 Å (1.0 nm) cutoff.`,
    );
  }

  return {
    trace,
    ssbonds,
    warnings,
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
  const ssbonds = filterCysSsbonds(prepared.trace.residues, prepared.ssbonds ?? prepared.trace.ssbonds);
  const decoded = decodeOutputs(output.s, output.v, prepared.n, { fixAtom });
  const R = buildAtoms(decoded, prepared.residueType, prepared.pos);
  const atoms = residueAtoms(R, prepared.residueType, prepared.trace.residues, ssbonds);
  const warnings = [...(prepared.warnings ?? prepared.trace.warnings ?? [])];
  const unknown = prepared.trace.residues.filter((_, i) => !atoms[i]).map((r) => `${r.chainId}${r.resNum}${r.insCode}:${r.inputName}`);
  if (unknown.length) {
    warnings.push(`cg2all has no template for ${unknown.length} residue(s) (${unknown.slice(0, 5).join(', ')}${unknown.length > 5 ? ', …' : ''}); their input atoms were kept`);
  }
  for (let i = 0; i < R.length; i++) {
    if (!Number.isFinite(R[i])) throw new Error('cg2all produced non-finite coordinates');
  }
  const { pdbText, atomCount } = toPdb({ ...prepared.trace, ssbonds }, atoms, { hydrogens, ligands, remarks });
  return { pdbText, atomCount, R, ss: decoded.ss, atoms, warnings };
}

/** One-call conversion with any engine exposing `predict(prepared)`. */
export async function convert(engine, pdbText, opts = {}) {
  const prepared = prepareInput(pdbText, { sequence: opts.sequence });
  const output = await engine.predict(prepared);
  return { prepared, ...assemble(prepared, output, opts) };
}
