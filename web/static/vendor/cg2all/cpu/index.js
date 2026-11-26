// cg2all pure-JS CPU fallback with the same `predict(prepared)` contract as the WebGPU engine.
// float64 throughout; matches PyTorch to ~1e-5 Å on the parity set but costs ~3.7 MFLOP per edge per layer.

import { buildNetwork, forwardCpu } from '../shared/model.js';

export * from '../shared/index.js';

export async function createCg2allCpuEngine({ weights }) {
  if (!weights?.tensors) throw new Error('cg2all: createCg2allCpuEngine needs loaded weights ({ manifest, tensors })');
  const net = buildNetwork(weights.tensors);
  return {
    backend: 'cpu',
    manifest: weights.manifest,
    fixAtom: Boolean(weights.manifest?.fixAtom),
    async predict(prepared) {
      const { s, v } = forwardCpu(net, prepared);
      return { s, v };
    },
    destroy() {},
  };
}
