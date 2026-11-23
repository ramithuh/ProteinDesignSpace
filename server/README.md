# Self-Hosting `denovo.run` (Zero-Python Host-GPU WebGPU Server)

Run **all 22 WebGPU protein-design models** (`AlphaFold 3`, `Boltz-2`, `RFD4`, `Genie 3`, `GNINA 1.3`, `ProteinMPNN`, `LigandMPNN`, `ThermoMPNN`, `ESM-C`, `Dyna-1`, `RFdiffusion`, `BoltzGen`, etc.) on your own GPU machine — with **zero Python, zero `uv`/`conda`, and zero CUDA wheel conflicts**.

Clients on any device (MacBook, iPad, phone, thin laptop, or another browser over LAN/Tailscale) can connect in **Host GPU (Remote)** or **Hybrid (Auto)** mode.

## How It Works

1. `server/selfhost-server.mjs` serves the built static `web/build/` app and launches **one persistent headless Chromium** pointed at its internal `/runner` route using the host machine's native GPU (Vulkan on Linux, Metal on macOS, D3D12 on Windows).
2. Model weights download once into the host's persistent Chromium OPFS/CacheStorage profile (`~/.cache/denovo-selfhost/profile`) and stay cached across restarts.
3. A strict FIFO GPU job queue serializes heavy GPU jobs across users so concurrent runs never OOM the host GPU, streaming live progress over Server-Sent Events (`/api/run-node`).
4. In the web UI, click the top-right **WebGPU** pill to switch between:
   - **Browser WebGPU** — everything runs locally in the viewer's browser
   - **Host GPU (Remote)** — every GPU block runs on the self-hosted machine's GPU
   - **Hybrid (Auto)** — lightweight models (`≤ 40 MB`, e.g. GNINA, ThermoMPNN, ProteinMPNN) stay instant in the viewer's browser while heavy models (`AF3`, `RFD4`, `Genie 3`, `Boltz-2`, etc.) or clients without WebGPU automatically route to the host GPU

## Quick Start

```bash
# 1. Build the static web app (Node 20+)
npm --prefix web ci
npm --prefix web run build

# 2. Install Playwright Chromium once (no Python required)
npm install --prefix server playwright-core
npx --prefix server playwright-core install chromium

# 3. Start the self-hosted server on port 8787
node server/selfhost-server.mjs
```

Open `http://<your-host>:8787` from any machine on your network or Tailnet — the UI automatically detects the same-origin host GPU server via `/api/health`.

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8787` | HTTP server port |
| `HOST` | `0.0.0.0` | Bind address |
| `DENOVO_API_TOKEN` | *(empty)* | Optional Bearer token required for `/api/*` endpoints |
| `DENOVO_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins (allows connecting public `https://denovo.run` to your private Tailscale GPU host) |
| `DENOVO_DATA_DIR` | `~/.cache/denovo-selfhost/profile` | Persistent host cache for model weights |
| `DENOVO_CHROME_PATH` | *(auto-detected)* | Explicit Chrome/Chromium binary path |
| `DENOVO_ALLOW_SOFTWARE_GPU` | `0` | Set to `1` to allow software WebGPU fallback on CPU-only hosts |
| `DENOVO_MAX_BODY_MB` | `128` | Max JSON request payload size in MB |
| `DENOVO_MAX_FOLD_RESIDUES` | `2048` | Default LocalFold (AF3 / Boltz / AF2 / ESMFold2) residue ceiling advertised to connected clients (512–4096) |

## API Endpoints

- `GET /api/health` — host GPU info, queue depth, active model worker
- `POST /api/run-node` — executes one GPU workflow node (`text/event-stream` progress + typed-array-preserving result)
- `POST /api/cancel` — cancels a queued or running job by `jobId` and releases VRAM
- `POST /api/release-gpu` — frees idle host GPU VRAM
