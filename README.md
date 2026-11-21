# ProteinDesignSpace (`denovo.run`)

A browser-native WebGPU node-graph workbench for de novo protein design, inverse folding, and single-sequence refolding. All inference runs locally on the user's GPU via WebGPU — zero server compute, zero conda environments, zero data upload.

## Capabilities

- **Visual DAG Workflow Canvas** (`@xyflow/svelte`): Build linear or branched protein design pipelines (e.g., comparing native RFD4 co-designed sequences against SolubleMPNN re-sequencing on one shared backbone), with typed color-coded ports, smart block insertion & auto-wiring (`+ Add Block` / `/`), and crossing-minimized auto-layout.
- **Browser-Native WebGPU Models** (sequential VRAM-isolated Dedicated Web Workers + OPFS weight caching):
  - **RFD4-Proteina (WebGPU INT8)** — all-atom sequence-structure co-design flow matching (unconditional, conditioned hotspots/motifs/H-bonds/CATH, partial diffusion, inverse folding, and 9 upstream benchmarks), powered by [`sokrypton/rfd4-proteina-webgpu`](https://github.com/sokrypton/rfd4-proteina-webgpu).
  - **Genie 3 (WebGPU FP16)** — fast 23M-parameter SE(3)-equivariant DDIM backbone diffusion + 1-step native SequenceNet, with weights streamed from [`ramithuh/genie3-webgpu`](https://huggingface.co/ramithuh/genie3-webgpu) on HuggingFace.
  - **ProteinMPNN / SolubleMPNN / Cα-MPNN (ONNX WebGPU)** — inverse folding with temperature and 3D-clickable fixed positions, powered by [`MurrellGroup/webProteinMPNN`](https://github.com/MurrellGroup/webProteinMPNN).
  - **LocalFold AF2 (WebGPU INT5)** — single-sequence AlphaFold2 monomer refolding producing all-atom PDBs, pLDDT, pTM, and PAE matrices locally, powered by [`sokrypton/localfold`](https://github.com/sokrypton/localfold).
- **Closed-Loop Designability & Intermediate Peek**:
  - Exact Kabsch SVD superposition (`det(R) = +1`), `scRMSD`, `scTM`, pLDDT, PAE heatmaps, and live Mol\* 3D superposition & diffusion trajectory playback.
  - Click any block, port label, or colored wire to peek at intermediate structures, sequences (with composition & recovery stats), folds, or compiled RFD4 `condition_spec` JSON — streaming live as each step finishes, or execute partial runs with **▶ Run to here**.
- **Shareable Recipes & Campaign Inspector**:
  - Compact `#p=` / `#g=` deflate-raw shareable URLs, live editable YAML/JSON recipe drawer, and a multi-round Campaign Inspector tab.

## Quick Start

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173` in a WebGPU-capable browser (Chrome, Edge, Arc, or Firefox 145+ on macOS).

## Tests & Production Build

```bash
cd web
npm test          # Vitest unit & upstream integration suite (offline)
npm run check     # svelte-check TypeScript + Svelte diagnostics
npm run build     # Syncs pinned upstreams & builds static SPA -> web/build/
npm run check:deploy  # Verifies required vendor engines & Cloudflare 25 MiB limit
```

## Deploying to Cloudflare Pages

Connect this GitHub repository in **Cloudflare Dashboard → Workers & Pages → Create → Pages → Connect to Git**:

| Setting | Value |
| --- | --- |
| **Production branch** | `main` |
| **Root directory (advanced)** | `web` |
| **Build command** | `npm run build && npm run check:deploy` |
| **Build output directory** | `build` |
| **Node.js version** | Auto-detected (`22`) from `web/.node-version` |

- Every push to `main` automatically deploys to production (`denovo.run`), and every feature branch / PR gets an isolated `*.pages.dev` preview URL.
- Model weights are **never** bundled in `web/build/` (~12.8 MiB total, max asset < 2 MiB): Genie 3 FP16 shards stream directly from HuggingFace (`ramithuh/genie3-webgpu`) and can be overridden at build time via `VITE_GENIE3_WEIGHTS_URL`.
- Static caching headers (`/_app/immutable/*` 1-year immutable, `/` & `/index.html` revalidate, `/vendor/*` SWR) are configured in `web/static/_headers`.

## Repository Layout

```text
ProteinDesignSpace/
├── tools/
│   └── sync-upstream.mjs       # Sparse sync of vendored upstream JS/WGSL runtimes
└── web/                        # SvelteKit 5 + TypeScript static SPA
    ├── src/lib/
    │   ├── engine/             # Scheduler, registry, graph-utils, peek, share, OPFS cache
    │   ├── protein-ops/        # PDB/mmCIF parsing, Kabsch SVD, geometry stats
    │   ├── workers/            # Dedicated WebGPU workers (rfd4, mpnn, localfold)
    │   ├── ui/                 # Canvas, block catalog, node inspector, peek & campaign UI
    │   ├── viewer/             # Headless Mol* superposition viewer & PAE heatmap
    │   └── __tests__/          # Vitest suite
    └── static/vendor/          # Vendored upstream headless JS/WGSL modules
```
