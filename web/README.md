# ProteinDesignSpace — web

SvelteKit playground app. All inference runs in the browser via a Web Worker using ONNX Runtime Web (WebGPU primary, WASM fallback, single-threaded).

## Status

V0 toolchain validator — loads the model, reports active EP, runs one forward pass at user-chosen `t`. No sampling loop or viewer yet (next milestone).

## Dev

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:5173>. Use Chrome / Edge / Arc for WebGPU; Firefox 145+ on macOS also works in stable.

### Model file

The page expects `/models/legacy-teacher/v1/model.onnx` to resolve. In dev this is a symlink at `web/static/models/legacy-teacher/v1/model.onnx → ../../../../../tools/exports/legacy-teacher/model.onnx`. Generate the artifact first:

```bash
cd ..
conda run -n legacy-model python tools/exports/legacy-teacher/export.py --opset 18
```

If the symlink is missing:

```bash
mkdir -p web/static/models/legacy-teacher/v1
ln -s ../../../../../tools/exports/legacy-teacher/model.onnx \
      web/static/models/legacy-teacher/v1/model.onnx
```

(The 855MB file is never copied or committed; it's served straight from the bundle directory.)

## Architecture in this slice

- `src/lib/worker/inference.worker.ts` — single-purpose worker. Loads ORT-Web with WebGPU EP, fetches & instantiates the model, runs forward.
- `src/lib/worker/protocol.ts` — typed message protocol between main thread ↔ worker.
- `src/routes/+page.svelte` — UI. Two buttons (load, forward), three inputs (length, seed, t), info panels.
- `src/routes/+layout.ts` — `ssr = false; prerender = true` so the static adapter emits a single SPA shell.

Next milestones (see repo root `README.md` for the full roadmap):

1. Port the Euler-FM sampler from `tools/exports/legacy-teacher/sample_onnx.py` into `src/lib/samplers/euler-fm-endpoint.ts`.
2. Lift the Mol\* trajectory viewer from `legacy-model-flow-map/examples/14_flow_map_trajectory_viewer.py` into `src/lib/viewer/`.
3. Wire generation → trajectory streaming → live viewer.
