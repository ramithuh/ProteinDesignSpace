# Adding a New Method to `denovo.run`

`denovo.run` runs protein design, structure prediction, and property prediction models **100% locally in the browser** via WebGPU (WGSL compute shaders or ONNX Runtime WebGPU/WASM) with weights streamed from Hugging Face and cached in OPFS.

If you want to add a new model block (backbone generator, inverse folder, structure predictor, or property/dynamics predictor), you can either follow the checklist below or hand the ready-made prompt to your favorite coding agent (Claude Code, Codex, Antigravity, Cursor, etc.).

---

## Copy-Paste Prompt for a Coding Agent

```text
Add a new browser-native WebGPU block for <MODEL_NAME> (<PAPER_OR_REPO_URL>) to denovo.run (ProteinDesignSpace).

Before writing code, inspect how existing WebGPU models are wired in this repo (for example `la-proteina`, `proteina`, `frameflow`, `dyna1`, and `thermompnn`) and follow the exact same patterns:

1. Runtime & Weights:
   - Never commit model weights to git; host sharded FP16/INT8 weights on Hugging Face and fetch/cache them via `web/src/lib/engine/opfs-cache.ts`.
   - Keep every static asset strictly under 25 MiB (Cloudflare Pages limit, verified by `npm run check:deploy`).
   - Vendor only minimal zero-dependency WebGPU/WGSL runtime files under `web/static/vendor/<model>/` (or use `onnxruntime-web/webgpu` via the vendored JSPI/WASM runtime in `web/static/vendor/ort/`).
   - Ensure all WGSL shaders are valid across Dawn (Chrome/Edge) and Naga (Firefox/Safari) — e.g. avoid unparenthesized negative literals like `x - -1.23` (`--` token error in Naga), keep storage buffers per stage <= 8 where possible, and stay within WebGPU workgroup/buffer limits.

2. Dedicated Web Worker:
   - Create `web/src/lib/workers/<model>.worker.ts` handling `{ type: 'run', id, inputs, params }` and `{ type: 'dispose', id }`.
   - Emit structured progress updates (`phase: 'weights' | 'infer'`, `progress: 0..1`, `detail`).
   - Output canonical typed artifacts (`StructureArtifact`, `SequenceSet`, `FoldedStructure`, `ResidueTrack`, `MutationScan`) with full provenance and zero NaNs.

3. Block Registration & UI Integration:
   - Register the adapter manifest, typed ports, defaults, and parameter schema in `web/src/lib/engine/registry.ts`.
   - Add the block to the appropriate palette category (`De Novo Generation`, `Sequence Design`, `Structure Prediction`, `Property Prediction`, or `Evaluation`) in `web/src/lib/engine/catalog.ts`.
   - Wire worker spawning/dispatch in `web/src/lib/engine/scheduler.ts`.
   - Add parameter validation & input preview support in `web/src/lib/engine/param-validation.ts` and `web/src/lib/engine/input-preview.ts`.
   - Add biological compatibility checks (e.g. Cα-only vs all-atom backbone, ligand/DNA support, monomer vs multimer) in `web/src/lib/engine/bio-capabilities.ts`.
   - Add a model card with license, weight size, and parity/verification notes in `web/src/lib/engine/model-cards.ts`.
   - Add a compact adapter code in `web/src/lib/engine/share.ts` so `#g=` share links stay short.

4. Verification (required before opening a PR):
   - Verify numerical parity against the reference PyTorch/JAX implementation (e.g. per-layer or end-to-end relRMS / RMSD / correlation on a deterministic seed).
   - Run `cd web && npx svelte-check --tsconfig ./tsconfig.json --threshold warning && npx vitest run && npm run build && npm run check:deploy`.
   - Run a real headless Chromium WebGPU end-to-end smoke test verifying weight fetch, GPU execution, 0 console/WGSL errors, and valid 3D/track/sequence output.
```

---

## Architecture Checklist (Where Each Piece Lives)

| Layer | File(s) | Purpose |
| :--- | :--- | :--- |
| **WebGPU runtime** | `web/static/vendor/<model>/` | Vendored zero-dep JS/WGSL inference code (no weights in git) |
| **Web Worker** | `web/src/lib/workers/<model>.worker.ts` | Isolated GPU execution, OPFS weight streaming, typed artifact outputs |
| **Block manifest** | `web/src/lib/engine/registry.ts` | Adapter ID, input/output port types, parameter schema & defaults |
| **Block palette** | `web/src/lib/engine/catalog.ts` | Category placement, search tags, and concise role description |
| **Execution** | `web/src/lib/engine/scheduler.ts` | Worker lifecycle, caching, cancellation, and progress streaming |
| **Bio & param checks** | `web/src/lib/engine/bio-capabilities.ts`, `param-validation.ts`, `input-preview.ts` | Static compatibility warnings (Cα vs all-atom, holo ligands, multimer) and pre-run preview |
| **Model card & sharing** | `web/src/lib/engine/model-cards.ts`, `share.ts` | License/attribution, parity notes, and compact `#g=` URL code |
| **AI Planner (optional)** | `web/src/lib/engine/workflow-planner.ts` | Optional Describe (`⌘K`) block-graph support |

## Reference WebGPU Ports

For clean, zero-dependency WebGPU/WGSL implementations and FP16/INT8 weight-sharding scripts, see:
- [`ramithuh/denovo-webgpu-models`](https://github.com/ramithuh/denovo-webgpu-models) — standalone WebGPU ports (RFdiffusion, BoltzGen, La-Proteina, Proteina, SiD-Protein, Genie 2, FrameFlow, FoldFlow 2, LigandMPNN, LASErMPNN, ThermoMPNN, Dyna-1, AntiFold/ESM-IF1, EvoDiff, etc.).

## Opening a PR

1. Fork this repository and create a feature branch.
2. Make sure `svelte-check`, `vitest run`, `npm run build`, and `npm run check:deploy` all pass cleanly.
3. Include your license attribution, weight repo URL, and a brief parity/smoke-test note in the PR description.
