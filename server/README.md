# Self-Hosted WebGPU Server for `denovo.run` (Zero Python)

Run **all 21 WebGPU models** (AlphaFold 3, Boltz-2, Protenix, RFD4, Genie 3, GNINA 1.3, ProteinMPNN, LigandMPNN, etc.) on a **remote GPU workstation or cluster** (RTX 5090, DGX Spark, HPC node) with **zero Python, zero `uv`/conda, and zero CUDA wheel conflicts**.

> **Do I need this on my own laptop?** No — if you're using Chrome/Edge on a Mac or PC with a GPU, just open `https://denovo.run` directly in **WebGPU (this computer)** mode. Self-hosting is primarily for offloading heavy jobs to a **remote GPU** (or using Firefox/Safari / Linux desktops where your daily browser lacks hardware WebGPU).

## 1. Build Once on the GPU Machine

```bash
# 1. Build the static web app
npm --prefix web ci
npm --prefix web run build

# 2. Install Playwright Chromium (headless WebGPU runtime)
npm install --no-save playwright-core
npx playwright-core install chromium
```

> **Note (Ubuntu 24.04 AppArmor):** If Playwright's bundled Chromium fails with `No usable sandbox!`, either pass your system Chrome (`DENOVO_CHROME_PATH=/usr/bin/google-chrome DENOVO_DISABLE_GPU_SANDBOX=1`) or run with `DENOVO_DISABLE_SANDBOX=1`.

## 2. Cluster / HPC Access (SSH Port Forwarding — Recommended)

Bind to loopback (`127.0.0.1`) on the GPU node with a token (so the server is never exposed to the LAN/internet and other users on the node cannot access it), then forward the port over SSH:

```bash
# On the GPU node
export DENOVO_API_TOKEN="dnv_$(openssl rand -hex 18)"
HOST=127.0.0.1 PORT=8790 DENOVO_API_TOKEN="$DENOVO_API_TOKEN" node server/selfhost-server.mjs

# On your laptop
ssh -L 8790:127.0.0.1:8790 user@gpu-node
```

Open **`http://127.0.0.1:8790`**, switch the top-right toggle to **Self-Hosted (remote machine)**, leave URL blank, paste your token once, and click **Sign in**. The server sets an **`HttpOnly; SameSite=Strict`** session cookie so refreshes stay signed in without storing your token in `localStorage`.

## 3. Tailscale HTTPS / Private Lab Server

Keep Node bound to `127.0.0.1` and terminate TLS with Tailscale Serve (or Caddy). Always set a high-entropy `DENOVO_API_TOKEN`; never expose plain `http://` directly on a public IP.

```bash
# Generate a strong Bearer token
export DENOVO_API_TOKEN="dnv_$(openssl rand -hex 18)"

# Keep the server bound to loopback + expose via Tailscale HTTPS
HOST=127.0.0.1 PORT=8790 DENOVO_API_TOKEN="$DENOVO_API_TOKEN" node server/selfhost-server.mjs
sudo tailscale serve --bg --https=8790 http://127.0.0.1:8790
```

> Binding `HOST=0.0.0.0` without `DENOVO_API_TOKEN` is rejected by default. Same-origin visitors sign in once via `POST /api/auth` (`HttpOnly; Secure; SameSite=Strict` cookie); cross-origin clients authenticate via `Authorization: Bearer <token>`.

## 4. Local Workstation Only (Optional)

If your daily browser lacks hardware WebGPU (e.g., Firefox, Safari, or Linux Chrome falling back to SwiftShader), you can also run the server locally on loopback and open `http://127.0.0.1:8787` in any browser:

```bash
node server/selfhost-server.mjs
```

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
