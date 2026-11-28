/**
 * Shared, zero-dependency building blocks for the ArtiosPotts ports.
 *
 * Everything here is pure ES modules with no runtime dependency and no WebGPU,
 * so the CPU reference (`potts/cpu`) and the WebGPU implementation
 * (`potts/webgpu`) can share it:
 *
 * * `constants.js` — restype vocabulary, element table, residue groupings and the
 *   organism AA frequency tables.
 * * `safetensors.js` — safetensors reader with FP16 / symmetric-per-row-INT8
 *   dequantization, plus the `manifest.json` weight-bundle loader.
 * * `featurize.js` — coordinate and graph featurizer: backbone N/CA/C/O plus the
 *   ProteinMPNN virtual CB, context heavy atoms, kNN graph, 16 Gaussian RBFs and
 *   the AF3 relative position encoding. No RDKit, no CCD, no structure library.
 * * `potts.js` — factorized Potts energy, conditional logits, pseudo-log-likelihood
 *   scoring and the alphabet/bias masking.
 * * `penalties.js` — complexity, enrichment, composition and net-charge penalties
 *   with hand-written gradients.
 * * `sampler.js` — steerable Gibbs / DLMC sampling from Potts parameters.
 *
 * Every symbol names the `artios` function it mirrors in its own doc comment.
 */

export * from './constants.js';
export * from './safetensors.js';
export * from './featurize.js';
export * from './potts.js';
export * from './penalties.js';
export * from './sampler.js';
