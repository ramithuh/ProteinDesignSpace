// cg2all WebGPU entry point: `createCg2allEngine({ device, weights })` → `{ predict(prepared), destroy() }`.
// `weights` is the `{ manifest, tensors }` result of shared/weights.js `loadWeights`.

import { buildNetwork } from '../shared/model.js';
import { createGpuEngine, QueuePacer, yieldToBrowser } from './engine.js';

export { QueuePacer, yieldToBrowser };
export * from '../shared/index.js';

export async function createCg2allEngine({ device, weights }) {
  if (!weights?.tensors) throw new Error('cg2all: createCg2allEngine needs loaded weights ({ manifest, tensors })');
  const net = buildNetwork(weights.tensors);
  const engine = await createGpuEngine({ device, net });
  return Object.assign(engine, { manifest: weights.manifest, fixAtom: Boolean(weights.manifest?.fixAtom) });
}
