# backbone/webgpu

Zero-dependency WebGPU (WGSL) port of `ArtiosBackbone`'s EDM diffusion module
(`artios/model/modules/diffusion_backbone.py :: DiffusionModule.forward`).

```js
import { createArtiosBackboneGpu, edmNoiseSchedule, edmStepNoise, edmSamplerUpdate } from './index.js';

const gpu = await createArtiosBackboneGpu({ device, weights, config, precision: 'fp16' });
const cond = await gpu.prepareConditioning(condFeatures);   // optional; a features object is cached too
const xDenoised = await gpu.denoiseStep(xNoisy, tHat, cond, { prevPred });
gpu.destroy();
```

## Inputs

- `weights`: `Map` or object of DiffusionModule `state_dict` tensors
  (`Float32Array` or `{ data, shape }`), or a `backbone/shared`
  `loadWeightBundle()` result. Keys may carry any prefix
  (`diffusion_module.`, `model.diffusion_module.`, ...).
- `config`: `DiffusionModuleConfig` fields, snake or camel case
  (`c_token, c_atom, c_s, c_z, c_timestep_embedding, num_blocks, num_heads`,
  nested `transformer` / `conditioning` accepted). Defaults are the production
  sizes (768 / 256 / 384 / 128 / 256, 16 blocks, 16 heads).
- `precision`: storage of linear weights, `'fp32' | 'fp16' | 'int8'`
  (packed `unpack2x16float` halves, or per-output-row symmetric int8).
  Accumulation is always FP32; layers with K < 32 stay FP32.
- `condFeatures`: the cached `DiffusionConditioning.compute_cache` outputs plus
  the atom layout, all flat row-major:
  `sBase [N*c_s]`, `z [N*N*c_z]` (after `transition_z`), `atomToTokenIdx [A]`,
  `tokenMask [N]`, `atomMask [A]`, `refCharge [A]`, `refElement [A*128]`,
  `refAtomNameChars [A*256]` (or `refFeatures [A*385]` already concatenated).
  snake_case keys are accepted, so a `backbone/cpu` `conditioningCache()`
  result (`{ sBase, z, features }`) can be passed directly.
- `xNoisy [S*A*3]`, `tHat` (number or `[S]`), optional `prevPred [S*A*3]`
  (the aligned previous prediction, in Å, for self-conditioning).

`denoiseStep` returns `r_pred = r_noisy * c_skip + F(c_in * r_noisy) * c_out`.
One EDM sampler step on top of it is `edmStepNoise` + `edmSamplerUpdate`
(see `generator.py :: EDMSampler`); the random rotation / translation
augmentation and the alignment of `prevPred` stay on the caller's side.

## Execution

- Per conditioning (once): pair bias of every block
  (`z * pair_mask -> z_layernorm -> z_proj`, chunked over query tokens),
  `linear_no_bias_f(ref features)`, CSR atom→token tables.
- Per shape: scratch buffers, uniforms and bind groups are built once; a step
  writes `x_noisy` and four per-sample EDM scalars, then replays the recorded
  dispatches (≈ 9 per DiT block) in paced submissions
  (`pacing: { submitBudgetFlops, maxDispatchesPerSubmit, maxInFlight }`).
- `destroy()` frees every buffer the engine created; the device is destroyed
  only if the engine requested it itself.

`LocalAttnTransformerGpu` (windowed atom attention from `transformer.py`) and
the individual kernels (`kernelOps`, `shaders`) are exported as building
blocks; `DiffusionModule` itself does not use local atom attention.

Tests: `node --test test/backbone-gpu.test.mjs` (headless Chromium, NVIDIA
Vulkan); `ARTIOS_GPU_QUICK=1` skips the full-size timing runs.
