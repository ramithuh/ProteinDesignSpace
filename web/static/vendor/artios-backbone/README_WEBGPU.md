# ArtiosBackbone — browser-native WebGPU / CPU runtime

Zero-dependency ES-module port of **ArtiosBackbone**'s all-atom EDM diffusion module,
vendored into `denovo.run` for the `artios-backbone` plugin (`shortCode: ab`).

| | |
| --- | --- |
| Upstream code | <https://github.com/Ligo-Biosciences/artios> (Apache-2.0) |
| Upstream weights | HF `Ligo-Biosciences/ArtiosBackbone` @ `3b8939e37aff1544cb74937bad3516673c9914ad`, `280000_ema_0.9999.pt` (546,762,475 B, SHA-256 `0016f6cf…b130fb`) |
| Port | `ramithuh/artios-webgpu` (`backbone/{shared,cpu,webgpu}`) |
| Browser weights | <https://huggingface.co/ramithuh/artios-webgpu> (`backbone/<precision>/manifest.json` + shards) |
| Exported module | `DiffusionModule` (`state_prefix = diffusion_module.`) — the released checkpoint contains nothing else |
| Parameters | 136,666,480 across 270 tensors |
| Architecture | `c_token 768 · c_atom 256 · c_s 384 · c_z 128 · c_timestep_embedding 256 · 16 AdaLN DiT blocks × 16 heads (head_dim 48) · sigma_data 16.0` |

## Layout

- `shared/index.js` — safetensors shard loader (`loadWeightBundle`), CIF/CCD parser
  (`parseCif`, `parseChemComp`, `parseMmcifAtoms`), AlphaFold3-style featurizer
  (`buildReferenceFeatures`, `buildTokenization`, `buildMotifFeatures`,
  `buildMotifDepthFeature`, `buildRasaFeature`), EDM Karras schedule + sampler
  (`edmNoiseSchedule`, `edmSamplerStep`, `edmSample`), Kabsch alignment
  (`kabschAlign`), `centreRandomAugmentation`, deterministic RNG (`createRng`).
- `cpu/index.js` — Float32 CPU reference denoiser (`createArtiosBackboneCPU`). Used
  for **both** the one-time timestep-independent `conditioningCache(features)` and as
  the CPU fallback denoiser (`denoise({ features, rNoisy, noiseLevel, cache, prevPred })`).
- `webgpu/{index,engine,shaders,quant}.js` — pure WGSL compute runtime
  (`createArtiosBackboneGpu({ device, weights, config, precision })` →
  `prepareConditioning(cache)` → `denoiseStep(xNoisy, tHat, cond, { prevPred })`).
  Per-conditioning work (pair bias of every block, atom reference-feature embedding,
  CSR atom→token tables) is computed once and replayed; `xNoisy` may be batched as
  `[n_sample, n_atom, 3]`.
- `LICENSE`, `NOTICE` — Apache-2.0 and the upstream attribution notice
  (Artios © 2025-2026 Ligo Biosciences; derivative of ByteDance Protenix).

Relative imports (`../shared/index.js`) are preserved, so the three directories must
stay siblings.

## Weight bundles (`artios-backbone-weights/1` manifests)

| Precision | Shards | Bytes | Size | Max per-tensor weight relRMS |
| --- | --- | --- | --- | --- |
| `fp32` | 23 | 546,699,056 | 521.4 MiB | — (reference) |
| `fp16` | 11 | 273,365,640 | 260.7 MiB | 2.24e-4 (`diffusion_conditioning.layernorm_t.weight`) |
| `int8` | 6 | 137,689,716 | 131.3 MiB | 2.07e-2 (`linear_no_bias_f.weight`, symmetric per-row + FP32 scales) |

Shard cap 25 MiB (26,214,400 B) per `.safetensors` file for Hugging Face / Cloudflare
Pages CDN compatibility. `loadWeightBundle` verifies every shard's SHA-256 against the
manifest and dequantizes to FP32 before the denoisers run.

## Verification (`ramithuh/artios-webgpu`)

- `test/real-oracle-gpu.test.mjs` — real checkpoint, real PyTorch FP32 oracle, hardware
  WebGPU on **NVIDIA GB10 (ARM64), Chromium headless (Dawn Vulkan)**:
  FP32 denoised-coordinate **relRMS 1.32e-6 cold** and **1.11e-6 with Kabsch-aligned
  self-conditioning**; FP16 and INT8 verified inside the 5e-3 / 5e-2 budgets.
- `test/backbone-gpu.test.mjs` — kernel-level and full-dimension checks plus warm
  timings. INT8 median `denoiseStep` **13.7 ms** at `N=256` tokens / `A=2048` atoms /
  `S=1` (≈ 2.7 s for a 200-step trajectory).
- `test/backbone-cpu.test.mjs` — CPU reference vs PyTorch intermediate-trace oracles and
  synthetic-weight structural invariants.

## `denovo.run` port mapping

| Port | Type | Notes |
| --- | --- | --- |
| in `structure` | `StructureArtifact` (optional) | motif spans + HETATM ligand; omitted ⇒ unconditional de novo design |
| out `structure` | `StructureArtifact` | multi-design `Collection` when `numDesigns > 1` |

The plugin lives in `web/src/lib/plugins/artios-backbone/`; the bundled CCD ideal
conformers for the 20 standard amino acids + `UNK` live in that folder
(`ccd-standard.ts`), and unknown ligand components are fetched from the RCSB CCD at run
time and parsed with `shared/parseChemComp`.
