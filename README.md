# ProteinDesignSpace

A browser-native playground for protein generative models. Inference runs locally on the user's GPU via WebGPU — no server, no shared compute, no conda environment to ship.

The closest analogues are LMSYS Arena and OpenAI Playground, but those are for LLMs. Nothing comparable exists for protein design models, even though "vibe-checking" (eyeballing what kind of folds, loops, and helices a model gravitates toward) is how researchers actually compare them — designability and diversity numbers miss it.

Repo is private until the playground ships publicly.

## Status

Phase: **foundation**. One model ported end-to-end as a pure-ONNX artifact; no UI yet.

- [x] legacy-teacher → ONNX export bundle (`tools/exports/legacy-teacher/`)
- [x] Numerical verification (max abs diff ~2e-5 vs PyTorch reference, L ∈ {50, 100, 150})
- [x] Standalone PDB generation from ONNX using only `numpy + onnxruntime`
- [ ] SvelteKit skeleton + Web Worker for inference
- [ ] Mol* viewer (lift from `legacy-model-flow-map/examples/14_flow_map_trajectory_viewer.py`)
- [ ] Euler-FM sampler in TypeScript
- [ ] Secondary structure / Rg / contact stats
- [ ] 2×4 synced trajectory grid
- [ ] Comparison view (model-A vs model-B, shared seed)
- [ ] Motif scaffolding (deferred to V2)
- [ ] Leaderboard — paper-reported stats + on-the-fly local stats (V3)

## Why this exists

1. **Vibe-checking is harder than it should be.** Comparing three methods today takes a day of dependency-pinning across conda environments that conflict.
2. **Sharing a generation means a screenshot or a colab.** No "send a link, see the structure" workflow.
3. **Distributions matter more than single samples.** Helix character, loop quality, fold diversity — only emerge from looking at many samples at once. Side-by-side visualization is the actual primitive.
4. **(Secondary, possibly bigger payoff)** — the byproduct of building this is a set of **portable `.onnx` artifacts** for protein design methods. There is no Hugging Face Hub equivalent for this corner of the field. Each model port is treated as a self-contained publishable bundle (export script + .onnx + metadata + verification test). The portability infra may be more valuable than the playground itself.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│             MAIN THREAD  (UI + Mol* viewers)               │
│  ┌────────────┐   ┌──────────────────────────────────┐    │
│  │ Controls   │   │ Mol* Grid (2×4 synced viewers)   │    │
│  │ - model    │   │ Trajectory slider + play         │    │
│  │ - length   │   └──────────────────────────────────┘    │
│  │ - NFE      │   ┌──────────────────────────────────┐    │
│  │ - seed     │   │ Stats Panel (SS%, Rg, contacts)  │    │
│  └────────────┘   └──────────────────────────────────┘    │
│                                         ▲ frame stream    │
└─────────────────────────────────────────│──────────────────┘
                                          │ postMessage
┌─────────────────────────────────────────│──────────────────┐
│             INFERENCE WORKER  (WebGPU)  │                  │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │ Model Adapter   │ → │ Sampler          │                │
│  │ (per-model TS)  │   │ (per-algorithm)  │                │
│  └─────────────────┘   └──────────────────┘                │
│              ↓                                              │
│  ┌────────────────────────────────────────────┐            │
│  │ ONNX Runtime Web (WebGPU EP / WASM fallback)│           │
│  └────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────┘
           ↑
┌────────────────────────────────────────────────────────────┐
│ Storage: OPFS (cached model files), localStorage (prefs)   │
│ Weights: self-hosted on blog server, versioned, immutable  │
└────────────────────────────────────────────────────────────┘
```

### Three load-bearing design rules

1. **Only the network forward pass crosses the ONNX boundary.** Samplers, schedules, per-step ops (CoM removal, frame computation) all stay in TypeScript. The sampling loop never appears in the `.onnx` file.
2. **Inference runs in a Web Worker.** Main thread stays responsive; trajectory frames stream out per sampling step.
3. **Adapters are thin per-model TS files; no unified abstraction.** Each adapter declares `(sampler, opts, buildForward closure)`. Unification happens one layer up, in the playground UI.

## Repository layout

```
proteindesignspace/
├── tools/
│   └── exports/                  # one self-contained bundle per ported model
│       └── legacy-teacher/
│           ├── export.py         # PyTorch ckpt → model.onnx
│           ├── verify.py         # numerical-match check
│           ├── sample_onnx.py    # numpy+onnxruntime reference sampler
│           ├── make_viewer.py    # generates standalone HTML viewer from a PDB
│           ├── metadata.json     # capabilities, defaults, I/O schema
│           ├── model.onnx        # the artifact (gitignored, ~855MB fp32)
│           └── samples/          # generated PDBs (gitignored)
├── src/                          # (TBD) SvelteKit playground app
│   ├── models/                   # adapters (one TS file per model)
│   ├── samplers/                 # by algorithm + output-type
│   ├── protein-ops/              # shared TS utilities (frames, SS, geometry, PDB)
│   ├── worker/                   # Web Worker entry + protocol
│   ├── viewer/                   # Mol* HTML components
│   └── ui/                       # Svelte components
└── README.md
```

## Adding a new model: the export bundle pattern

Each ported model lives in its own directory under `tools/exports/` and is *publishable independently of the playground*. Someone with the bundle alone can verify and run the model in five minutes.

| File | Purpose |
|---|---|
| `export.py` | Loads the PyTorch ckpt; wraps the model with a thin `*Export` module that flattens dict inputs to plain tensors and zeros out unused conditioning paths; traces to ONNX with dynamic axes. |
| `verify.py` | Runs the PyTorch reference and the ONNX runtime on seeded inputs; asserts max abs diff < 1e-4. Sweeps a few lengths to confirm dynamic axes work. |
| `sample_onnx.py` | End-to-end sampler with only `numpy + onnxruntime` deps. Proves the artifact is self-contained. Doubles as the 1:1 reference for the TS sampler. |
| `metadata.json` | Capabilities, default sampler config, I/O schema, expected size. Consumed by the playground's model registry. |
| `README.md` | What the method is, how to export, how to verify. |
| `.gitignore` | Ignores `*.onnx`, `*.onnx.data`, `samples/`. |

### legacy-teacher port: lessons captured

Things future-you will be glad you wrote down:

- **Lightning prefix strip.** State-dict keys start with `model.` — strip before `load_state_dict`. See `_strip_lightning_prefix` in `export.py`.
- **Use the legacy ONNX exporter** (`dynamo=False`). PyTorch 2.11's default FX-based dynamo exporter does not correctly propagate dynamic batch dimensions through `tensor.view(*x.shape[:-1], heads, head_dim)` — it bakes in the literal batch size from the example input. The legacy TorchScript-based exporter handles this fine and is still supported (with a deprecation warning).
- **Export at opset 18, not 17.** torch's automatic 18→17 conversion leaves an opset-18 `num_outputs` attribute on `Split` nodes that ONNX Runtime rejects. Just export at 18 directly.
- **`backbone_aa_idx` is DCE'd by the legacy exporter.** It's only referenced inside the `if pair_dim > 0:` branch, and we set `pair_dim=0`. The ONNX has 5 inputs, not 6. `verify.py` and `sample_onnx.py` filter inputs to match what `session.get_inputs()` reports.
- **Frozen constants from legacy-model/data**: `BACKBONE_ATOM_IDS = [18, 2, 1]` (N, CA, C in RF2AA encoding); `ALA_TOKEN = 1` (raw index 0 + padding shift). Inlined in `sample_onnx.py` with provenance.
- **Numerical match**: observed max abs diff ~2e-5 against fp32 PyTorch reference across L ∈ {50, 100, 150}, threshold 1e-4.
- **Sampler**: forward Euler over x̂_1 endpoint prediction (head named `backbone_coords_velocity` but returns x̂_1). Linear update `x + dt·(x̂_1 − x)/(1 − t)`, with CoM zero-centering per step, and last-step snap to x̂_1 to avoid the `1/(1−t)` blowup. Schedule: `log` with `p=2.0`. NFE=100 for the playground; 400 is the user's production setting.
- **CPU baseline latency**: ~2.6 s/step, ~65 s/sample at L=100. WebGPU expected to be 10–30× faster based on SAM2 / WebLLM benchmarks.

## Samplers

Samplers live in `src/samplers/` and are organized by **algorithm family + model output type**, not per-model. Most of the protein design literature is covered by ~5–8 sampler files:

| File | Algorithm | Output type | Models |
|---|---|---|---|
| `euler-fm-endpoint.ts` | Forward Euler ODE | x̂_1 prediction | legacy-teacher |
| `sde-proteina-endpoint.ts` | Proteina-style SDE with g(t) | x̂_1 | legacy-model-student, FrameFlow |
| `euler-fm-velocity.ts` | Forward Euler ODE | v(x, t) | (future) |
| `heun-fm-endpoint.ts` | 2nd-order predictor-corrector | x̂_1 | (future low-NFE) |
| `ddpm-ancestral-noise.ts` | DDPM discrete reverse process | ε prediction | (future) Genie2, Chroma |
| `ddim-noise.ts` | DDIM deterministic | ε | (future) |

**Decision rule for new samplers:** if you'd cite a different paper for the change, it's a new sampler. If you'd cite the same paper but tune a hyperparameter, it's an opt on an existing sampler. SDE-Proteina with γ=0.1 vs γ=0.2 → same sampler, different opts. SDE-Proteina vs SDE-VP → different samplers.

**Schedules** are pure functions in `src/protein-ops/schedules.ts` (`logSchedule(p)`, `uniformSchedule()`, `cosineSchedule()`, etc.). Passed as opts to samplers; never bundled inside them.

**Per-step ops** (`zeroCenterCoM`, `frenetFrames`, `symmetryProject`) live in `src/protein-ops/`. Also passed as opts.

**Shape of a sampler** (for reference):

```typescript
// samplers/sde-proteina-endpoint.ts
export interface SDEProteinaOpts {
  numSteps: number;
  schedule: (n: number) => Float32Array;
  postStep?: (x: ort.Tensor) => ort.Tensor;
  gamma: number;
  gtMax: number;
  seed: number;
  onFrame?: (x: ort.Tensor, t: number) => void;
}

export async function sampleSDEProteinaEndpoint(
  forward: (x: ort.Tensor, t: number) => Promise<ort.Tensor>,
  xInit: ort.Tensor,
  opts: SDEProteinaOpts,
): Promise<ort.Tensor> { /* ... */ }
```

The `forward` callback is the *only* thing the adapter provides — a closure over the ONNX session and the static features. The sampler stays model-blind.

## Deployment

- **Domain**: `playground.ramith.fyi` (subdomain, no iframe complexity).
- **Stack**: SvelteKit static build.
- **Inference**: ONNX Runtime Web with WebGPU EP primary, single-threaded WASM fallback. No SharedArrayBuffer / COOP+COEP headers (we deliberately skip threaded WASM to avoid the cross-origin-isolation dance).
- **Weight hosting**: self-hosted on the blog server. Versioned paths (`/models/<id>/v<n>/model.onnx`), `Cache-Control: immutable`, gzip/brotli compression, HTTP range requests enabled.
- **Blog integration**: blog posts link out to deep-linkable playground URLs (e.g., `playground.ramith.fyi/?model=legacy-model&length=100&seed=42`). No iframe.

## Things we are explicitly NOT doing

- **No custom WGSL kernels.** ONNX Runtime Web's WebGPU EP handles every op our models need. Custom kernels become worthwhile only if profiling shows ORT-Web is the bottleneck on a specific op (e.g., triangle attention at long sequences). Reserve ORT-Web's JSEP custom-op hook as the escape hatch.
- **No unified `Tensor` abstraction across samplers.** Use `ort.Tensor` and `Float32Array` directly.
- **No plugin loader** for "drop a folder, get a model." Just edit `src/models/registry.ts`.
- **No server-side inference path.** (The `.onnx` files would deploy server-side trivially via `onnxruntime-gpu`, but it's not the immediate target.)
- **No in-browser designability scoring.** ProteinMPNN ports easily; the fold model (ESMFold 3B) is too large for browser. Designability is a phase-2 problem.
- **No RFdiffusion port.** Built on `e3nn` (custom tensor-product autograd) — not browser-tractable without rewriting equivariant layers from scratch. Skip unless someone open-sources an e3nn-free reimplementation.

## Roadmap

- **V0** (in progress): single-model playground. legacy-teacher, unconditional generation, 2×4 synced viewers, SS/Rg/contact stats. The minimum useful version.
- **V1**: side-by-side comparison. Same seed, two columns, two model variants. The "I want to A/B my new flow matching method against my last checkpoint" version.
- **V2**: motif scaffolding (conditioning UI + adapter input changes).
- **V3**: leaderboard (paper-reported numbers + on-the-fly client-computed stats).
- **V4+**: additional model ports — Genie2 (triangle ops, DDPM sampler), FrameFlow (IPA, SDE-Proteina), Chroma. ProteinMPNN if/when designability becomes worth pursuing.
