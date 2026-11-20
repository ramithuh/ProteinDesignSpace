# `denovo.run` — Web App

SvelteKit 5 static SPA (`@sveltejs/adapter-static`) running browser-native WebGPU protein design and refolding pipelines inside Dedicated Web Workers with OPFS / CacheStorage weight caching.

## Development

```bash
npm install
npm run dev
```

Open <http://localhost:5173> in a WebGPU-enabled browser (Chrome, Edge, Arc, or Firefox 145+ on macOS). Model weights are fetched directly from their canonical upstream Hugging Face / CDN sources on first run and cached locally in browser storage.

## Testing & Verification

```bash
npm test          # Run Vitest unit & integration tests
npm run check     # Run svelte-kit sync + svelte-check
npm run build     # Build static bundle into ./build
```

## Architecture

- `src/lib/engine/scheduler.ts` — Topological DAG scheduler with content-addressed memoization, live intermediate output streaming, and a single-active-GPU-worker VRAM mutex (`dispose` → `terminate`).
- `src/lib/workers/` — Isolated Dedicated Web Workers for `rfd4-proteina`, `web-protein-mpnn`, and `localfold-af2`.
- `src/lib/protein-ops/` — Pure TypeScript PDB/mmCIF parser, Kabsch SVD alignment, TM-score, and backbone geometry statistics.
- `src/lib/ui/` & `src/lib/viewer/` — Interactive `@xyflow/svelte` DAG canvas, searchable `+ Add Block` catalog, tabbed `Parameters` / `Peek` inspector, Mol\* 3D superposition viewer, PAE plot, and Campaign Inspector.
