# Adding a New Method to `denovo.run`

`denovo.run` runs protein design, structure prediction, and property prediction models **100% locally in the browser** via WebGPU (WGSL compute shaders or ONNX Runtime WebGPU/WASM) with weights streamed from Hugging Face and cached in OPFS / CacheStorage.

With the **3-layer Unifying Engine** (`Layer 1: Core Kernel` → `Layer 2: Domain Packs` → `Layer 3: Model Plugins`), adding a new model requires creating **one self-contained plugin folder** (`web/src/lib/plugins/<id>/`) and adding **one registration line** in `web/src/lib/plugins/index.ts`. The scheduler, share/YAML codec, preflight estimator, peek inspector, model card popover, and AI workflow planner (`⌘K`) all discover your plugin automatically.

---

## Copy-Paste Prompt for a Coding Agent

```text
Add a new browser-native WebGPU ModelPlugin for <MODEL_NAME> (<PAPER_OR_REPO_URL>) to denovo.run (ProteinDesignSpace).

Before writing code, inspect existing Layer 3 plugins under `web/src/lib/plugins/` (for example `dyna1/`, `thermompnn/`, `gnina/`, `genie3/`, or `pdb-input/` for CPU plugins) and follow the exact same patterns:

1. Runtime & Weights:
   - Never commit model weights to git (CI fails via `npm run check:deploy` if any `web/static/vendor/*/weights` directory exists). Host sharded FP16/INT8 weights on Hugging Face and stream/cache them via `web/src/lib/engine/opfs-cache.ts` (declaring every CacheStorage / OPFS namespace in `plugin.cachePolicy`).
   - Keep every static asset strictly under 25 MiB (Cloudflare Pages limit, verified by `npm run check:deploy`).
   - Vendor only minimal zero-dependency WebGPU/WGSL runtime files under `web/static/vendor/<model>/` (or use `onnxruntime-web/webgpu` via `web/static/vendor/ort/`).
   - Ensure all WGSL shaders are valid across Dawn (Chrome/Edge) and Naga (Firefox/Safari) — avoid unparenthesized negative literals like `x - -1.23`, keep storage buffers per stage <= 8 where possible, and stay within WebGPU limits.

2. Cohesive Layer 3 Plugin Folder (`web/src/lib/plugins/<id>/`):
   - Create `web/src/lib/plugins/<id>/<id>.worker.ts` wrapped in `createWorkerHarness()` from `$lib/kernel/worker-harness`.
     - Implement `init(runtimeCtx)` (passing `runtimeCtx.signal` into fetches), `run(ctx)` (checking `ctx.signal.aborted` between steps), `abort?(runId)`, and `dispose()`.
     - Only include `transferables` on buffers freshly owned by the run (never cached buffers).
     - Emit canonical typed artifacts (`StructureArtifact`, `SequenceSet`, `FoldedStructure`, `ResidueTrack`, `MutationMatrix`, `TrajectoryFrames`, `SequenceEmbedding`, etc.) with provenance and zero NaNs.
   - Create `web/src/lib/plugins/<id>/index.ts` exporting a `ModelPlugin`:
     - `id`, `domainId: PROTEIN_DOMAIN_ID`, `manifest` (typed `inputs`, `outputs`, `modes`, `paramSchema`, `secondaryStages` for multi-task/omni models),
     - `execution` (`runtime`, lazy `workerLoader: () => new MyWorker()`, `supportsCooperativeAbort`, `abortGraceMs`, `vramLeaseMode`),
     - `cachePolicy` (every CacheStorage / OPFS bucket used so Clear Cache stays complete),
     - `share` (unique `shortCode` for compact `#g=` links, `stripOnShareKeys`, optional `prepareForShare` / `restoreOmittedInlineAsset`),
     - `catalog` (`categoryId`, `conciseRole`, `searchKeywords`),
     - `display` (`formatCanvasSummary`, optional `resolveSlots` for mode-conditional outputs & viewport roles),
     - `preflight` (VRAM/download/work-item estimator),
     - `modelCard`, `guards`, and `planner` (`plannerRoles`, keywords, capability tags for automatic `⌘K` AI planner discovery).
   - Register the plugin with ONE line in `web/src/lib/plugins/index.ts`.

3. Adding a New Scientific Domain (Optional):
   - Register a Layer 2 `DomainPack` under `web/src/lib/domains/<domain>/` defining custom `stages`, `portTypes` (with `summarize` and `formatNodeFooterKpi`), `paramFieldKinds`, `artifactViewers`, and `onboardingTour` — zero changes to Layer 1 (`src/lib/kernel/`).

4. Verification (required before opening a PR):
   - Verify numerical parity against the reference PyTorch/JAX implementation on a deterministic seed.
   - Run:
     `cd web && npx svelte-check --tsconfig ./tsconfig.json --threshold warning && npx vitest run && npm run build && npm run check:deploy`
   - Run a real headless Chromium WebGPU end-to-end smoke test verifying weight fetch, GPU execution, 0 console/WGSL errors, and valid output.
```

---

## Architecture Map (3-Layer Unifying Engine)

| Layer | Directory / File | Responsibility |
| :--- | :--- | :--- |
| **Layer 1 · Core Kernel** | `web/src/lib/kernel/` | Domain-agnostic DAG/plugin contracts (`PluginRegistry`, `createWorkerHarness`, `compare-kernel`, `planner-catalog-index`, `iterative-loop`, `condition-intent`). Zero protein-domain imports. |
| **Layer 2 · Domain Pack** | `web/src/lib/domains/protein-biopolymer/` | Shared protein/biopolymer `PortType`s, stages, benchmarks, preflight summaries, and onboarding tour metadata. Zero plugin imports. |
| **Layer 3 · Model Plugins** | `web/src/lib/plugins/<id>/` | Self-contained model folder (`index.ts` + `<id>.worker.ts` or `cpu-runner.ts`) registered in `web/src/lib/plugins/index.ts`. |
| **WebGPU Vendor Runtime** | `web/static/vendor/<model>/` | Vendored zero-dep JS/WGSL inference code (never weights; checked dynamically by `tools/check-deploy.mjs`). |
| **Engine Runtime** | `web/src/lib/engine/` | `ExecutionScheduler`, `#g=`/YAML `share.ts`, `run-preflight.ts`, `peek.ts`, `workflow-planner.ts` — all dispatch via `PluginRegistry`. |
| **UI Shell** | `web/src/lib/ui/` | Modular Svelte 5 components (`InspectorHeader`, `DockingScoreCard`, `SchemaParamFields`, `ArtifactPeekPanel`, `WorkflowCanvas`). |

## Reference WebGPU Ports

For clean, zero-dependency WebGPU/WGSL implementations and FP16/INT8 weight-sharding scripts, see:
- [`ramithuh/denovo-webgpu-models`](https://github.com/ramithuh/denovo-webgpu-models) — standalone WebGPU ports (RFdiffusion, BoltzGen, La-Proteina, Proteina, SiD-Protein, Genie 2, FrameFlow, FoldFlow 2, LigandMPNN, LASErMPNN, ThermoMPNN, Dyna-1, AntiFold/ESM-IF1, EvoDiff, etc.).

## Opening a PR

1. Fork this repository and create a feature branch.
2. Make sure `svelte-check`, `vitest run` (including all contract & locality tests), `npm run build`, and `npm run check:deploy` all pass cleanly.
3. Include your license attribution, weight repo URL, and a brief parity/smoke-test note in the PR description.
