# Self-Hosted WebGPU Server for `denovo.run` (Zero Python)

Run every WebGPU model in `denovo.run` on a **remote GPU workstation or cluster** (RTX 5090, DGX Spark, HPC node) with **zero Python, zero `uv`/conda, and zero CUDA wheel conflicts**.

> **Do I need this on my own laptop?** No — if you're using Chrome/Edge on a Mac or PC with a GPU, just open [`https://denovo.run`](https://denovo.run) directly in **WebGPU (this computer)** mode. Self-hosting is for offloading heavy jobs (and >512 aa folds) to a **remote GPU**, or using a browser without hardware WebGPU.
>
> **Fast-moving repo — pull regularly:** This repository changes quickly (new models, WebGPU kernel speedups, and bug fixes land frequently). When self-hosting, periodically run `git pull && npm --prefix web ci && npm --prefix web run build` and restart `node server/selfhost-server.mjs` so your server stays in sync with [`https://denovo.run`](https://denovo.run) (see [Updating Later](#updating-later) below — your cached model weights are preserved).

## Prerequisites (on the GPU Machine)

- **Node.js 22+** (`node -v`) — install via `nvm`, system package manager, or a standalone Node tarball (no root required on HPC).
- **GPU drivers with Vulkan** (Linux NVIDIA/AMD), Metal (macOS), or Direct3D 12 (Windows). You do **not** need a desktop GUI or monitor on the GPU machine — the server runs headless Chromium under the hood, and you view the UI in your regular browser on your laptop.

## 1. Clone & Build Once on the GPU Machine

```bash
git clone https://github.com/ramithuh/ProteinDesignSpace.git
cd ProteinDesignSpace

# Build the static web app -> web/build/
npm --prefix web ci
npm --prefix web run build

# Install Playwright Chromium (headless WebGPU runtime for the server)
npm install --no-save playwright-core
npx playwright-core install chromium
```

> **Model weights:** Weights are **not** downloaded during install. The first time you run a model, the server downloads its sharded safetensors once from Hugging Face into `DENOVO_DATA_DIR` (default `~/.cache/denovo-selfhost/profile`) and reuses them across all future browser sessions and restarts.

## 2. Start on the GPU Machine

Pick **Option A (HPC / Cluster via SSH)** or **Option B (Private Workstation via Tailscale HTTPS)**. Both keep Node bound privately to `127.0.0.1` so it is never exposed to the public internet or LAN.

### Option A: Cluster / HPC Access (SSH Port Forwarding — Recommended)

```bash
# On the GPU node
export DENOVO_API_TOKEN="dnv_$(openssl rand -hex 18)"
echo "Your API token: $DENOVO_API_TOKEN"
HOST=127.0.0.1 PORT=8790 DENOVO_API_TOKEN="$DENOVO_API_TOKEN" node server/selfhost-server.mjs
```

Then, in a terminal **on your laptop**:

```bash
ssh -L 8790:127.0.0.1:8790 user@gpu-node
```

Open **`http://127.0.0.1:8790`** in your laptop browser.

### Option B: Private Lab Workstation (Tailscale HTTPS)

```bash
# On the GPU workstation
export DENOVO_API_TOKEN="dnv_$(openssl rand -hex 18)"
echo "Your API token: $DENOVO_API_TOKEN"
HOST=127.0.0.1 PORT=8790 DENOVO_API_TOKEN="$DENOVO_API_TOKEN" node server/selfhost-server.mjs

# In another terminal (persists across reboots)
sudo tailscale serve --bg --https=8790 http://127.0.0.1:8790
```

Open the `https://<machine>.<tailnet>.ts.net:8790` URL printed by `tailscale serve`.

> Binding `HOST=0.0.0.0` without `DENOVO_API_TOKEN` is rejected by default. Never expose plain `http://` directly on a public IP.
>
> **Connecting from `https://denovo.run` instead of opening the server URL directly?** Pass `DENOVO_ALLOWED_ORIGINS=https://denovo.run` when starting `selfhost-server.mjs`.

## 3. Connect in Your Browser

1. Open the URL from Step 2 (`http://127.0.0.1:8790` for SSH, or your `https://...ts.net:8790` Tailscale URL).
2. Click **`Self-Hosted (remote machine)`** in the top-right header toggle.
3. Leave **Host Server URL** blank (`same origin`), paste your `DENOVO_API_TOKEN` once, and click **Sign in**.
4. The server verifies your token and sets an **`HttpOnly; SameSite=Strict`** session cookie (30 days) — refreshes and new tabs stay signed in automatically without storing the token in `localStorage`.

> Lightweight CPU blocks and the 3D Mol\* viewer always run locally in your browser; all GPU blocks automatically stream to your remote GPU.

## Updating Later

Because `ProteinDesignSpace` evolves rapidly, pull and rebuild whenever you want the latest models, kernel optimizations, and bug fixes:

```bash
cd ProteinDesignSpace
git pull
npm --prefix web ci
npm --prefix web run build
```

Then **restart `node server/selfhost-server.mjs`** so both the HTTP server and its persistent headless Chromium `/runner` tab load the newly built bundle. Your cached model weights in `DENOVO_DATA_DIR` (default `~/.cache/denovo-selfhost/profile`) are preserved across updates.

## Troubleshooting

- **Ubuntu 24.04 AppArmor (`No usable sandbox!`):** Playwright's bundled Chromium may be blocked by Ubuntu 24.04 user-namespace AppArmor rules. Either point to system Chrome:
  ```bash
  DENOVO_CHROME_PATH=/usr/bin/google-chrome DENOVO_DISABLE_GPU_SANDBOX=1 \
    HOST=127.0.0.1 PORT=8790 DENOVO_API_TOKEN="$DENOVO_API_TOKEN" node server/selfhost-server.mjs
  ```
  or pass `DENOVO_DISABLE_SANDBOX=1` inside containers/HPC nodes without user-namespace sandboxing.
- **Software fallback (`SwiftShader`) on Linux:** Make sure NVIDIA/AMD Vulkan ICD libraries are installed (`ls /usr/share/vulkan/icd.d/`). The server refuses SwiftShader by default unless `DENOVO_ALLOW_SOFTWARE_GPU=1` is set.
- **Custom weight cache location (e.g. HPC scratch/work storage):** Set `DENOVO_DATA_DIR=/cluster/work/$USER/denovo-profile`.

## Optional: Local Workstation Only

If your daily browser lacks hardware WebGPU (e.g., Safari, Firefox, or Linux Chrome falling back to SwiftShader), you can run `node server/selfhost-server.mjs` locally and open `http://127.0.0.1:8787` in any browser.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8787` | HTTP listen port |
| `HOST` | `127.0.0.1` | Bind address (non-loopback requires `DENOVO_API_TOKEN`) |
| `DENOVO_API_TOKEN` | *(empty)* | Bearer token required for `/api/*` endpoints (verified in constant time) |
| `DENOVO_ALLOWED_ORIGINS` | *(same-origin only)* | Comma-separated explicit origins (e.g. `https://denovo.run`) allowed for cross-origin API calls |
| `DENOVO_DATA_DIR` | `~/.cache/denovo-selfhost/profile` | Persistent host cache for model weights |
| `DENOVO_CHROME_PATH` | *(auto-detected)* | Explicit Chrome/Chromium binary path |
| `DENOVO_GPU_LABEL` | *(auto-detected)* | Optional display name override for the GPU pill |
| `DENOVO_ALLOW_SOFTWARE_GPU` | `0` | Set to `1` to allow SwiftShader software WebGPU fallback |
| `DENOVO_DISABLE_SANDBOX` | `0` | Set to `1` only in containers lacking Chromium user-namespace sandboxing |
| `DENOVO_MAX_BODY_MB` | `64` | Max JSON request payload size in MiB (1–256) |
| `DENOVO_MAX_QUEUE_DEPTH` | `8` | Max queued GPU jobs before returning HTTP `429` (1–64) |
| `DENOVO_JOB_TIMEOUT_MS` | `600000` | Per-job wall-clock deadline in ms (default 10 min) |
| `DENOVO_MAX_FOLD_RESIDUES` | `2048` | Server-enforced LocalFold residue ceiling (512–4096) |

## API Endpoints

- `POST /api/auth` — exchanges `{ token }` for a 30-day `HttpOnly; SameSite=Strict` HMAC session cookie (rate-limited per IP; sets `Secure` on HTTPS/non-loopback)
- `POST /api/logout` — clears the `HttpOnly` session cookie
- `GET /api/health` — host GPU hardware info, busy state, queue depth, `maxFoldResidues`, `authenticatedBy`
- `POST /api/run-node` — executes one validated GPU workflow node (`Content-Type: application/json`, returns `text/event-stream` starting with `event: started` containing `{ jobId, cancelToken }`)
- `POST /api/cancel` — cancels a queued or running job using its owner `{ jobId, cancelToken }`
- `POST /api/release-gpu` — frees idle host GPU VRAM
