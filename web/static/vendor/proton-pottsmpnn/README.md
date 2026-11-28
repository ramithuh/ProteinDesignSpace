# proton-pottsmpnn-webgpu

A complete, zero-dependency JavaScript + WebGPU (WGSL) numerical inference port of
**Proton-PottsMPNN** — a pH-aware PottsMPNN that scores protein backbones in an
explicit protonation-state vocabulary and designs pH-selective sequences.

Nothing in `shared/`, `cpu/` or `webgpu/` imports anything: they are pure ES modules
plus WGSL. The only Python is in `tools/`, and it runs once, offline, to export the
checkpoint and to dump the PyTorch oracle the tests check against.

```
shared/index.js    v6 vocabulary · safetensors (FP32/FP16/INT8) · PDB parser ·
                   kNN/RBF featurizer · Potts energies · pH-selective block descent
cpu/index.js       pure-JS backend, matched to the PyTorch FP32 oracle
webgpu/            WGSL kernels (shaders.js), GPU-resident engine (engine.js), API (index.js)
tools/             export_weights.py · dump_oracle.py
test/              cpu.test.mjs · gpu.test.mjs (headless Chrome on the real adapter)
```

## What was ported

| Piece | Status |
| --- | --- |
| Backbone featurizer: N/CA/C/O + virtual CB, CA kNN graph, 25×16 atom-pair RBFs, chain-aware relative positional encoding | ✅ exact |
| `PottsMPNN` encoder: 3 `EncLayer`s (node update, feed-forward, edge update) | ✅ exact |
| Potts head: `etab_out`, edge masking, diagonal self edge, reciprocal merge | ✅ exact |
| Potts readouts: `E_global`, per-position conditional energies, exact mutation `ΔH` | ✅ exact |
| v6 protonation vocabulary (30 tokens: `HIS-P/S/A`, `ASP-P/D/A`, `GLU-P/D/A` + 21 standard) | ✅ exact |
| pH-selective gap `E_P − E_D` | ✅ exact |
| pH-selective block-descent optimizer `(1−λ)·z(E_global) + λ·z(Σ(E_P−E_D))` | ✅ exact |
| Decoder: 3 `DecLayer`s, all four causality patterns, teacher forcing + autoregressive sampling | ✅ exact |
| HBPLUS / propka / FLAML protonation labeller | ❌ out of scope — see below |

The GPU backend runs the encoder, the Potts head and the teacher-forced decoder as
WGSL kernels; the featurizer, the Potts energies and the optimizer are the shared
`shared/` code on both backends, because they are index-heavy and sequential rather
than dense arithmetic.

The one piece of the reference optimizer intentionally left at its defaults is the
sequence-local density machinery (`adjacent_repeat_weight`,
`repetitive_window_*`): it is implemented and exposed as options, but it is off in
the reference's own default criteria, so the oracle does not exercise it.

### Protonation states without the labeller

The reference pipeline assigns each titratable residue's state with an external
HBPLUS run and a FLAML/propka classifier. That is deliberately **not** ported: those
are an external C binary and Python pickles, neither of which belongs in a browser.

Instead the featurizer assigns tokens from the PDB residue names:

* non-titratable residues keep their three-letter code;
* titratable residues default to their **physiological state** at pH ≈ 7.4 —
  `ASP → ASP-D`, `GLU → GLU-D`, `HIS → HIS-S` (`PHYSIOLOGICAL_STATE`);
* any position can be **pinned explicitly** through `overrides`, keyed by
  0-based position or by a `"<chain><resSeq>"` label:

```js
const structure = structureFromPDB(pdbText, {
  overrides: { A42: 'HIS-P', 7: 'GLU-P' },   // this is the labeller's job, done by hand
});
```

Pinning a *centre* for the selective objective is separate from the input token and
is what `resolveCentres` + the optimizer's `centres` option are for.

## Quick start

```bash
# once: export the checkpoint and dump the oracle (needs torch + the reference tree)
python tools/export_weights.py          # -> .cache/weights/{manifest.json,*.safetensors}
python tools/dump_oracle.py             # -> fixtures/oracle.json + the backbone fixture

npm run test:cpu
npm run test:gpu
```

### CPU

```js
import { createProtonPottsCPU } from 'proton-pottsmpnn-webgpu/cpu';

const model = await createProtonPottsCPU({ baseUrl: '.cache/weights', precision: 'fp32' });
const structure = model.structureFromPDB(pdbText);
const out = model.run(structure);              // featurize + encoder + Potts head

out.h            // [L, V]          single-body field
out.J            // [L, K, V, V]    couplings; J[i,0] is diagonal and its diagonal is h[i]
out.energy       // E_global = H(s)
model.destroy();
```

### WebGPU

```js
import { createProtonPottsWebGPU } from 'proton-pottsmpnn-webgpu';

const model = await createProtonPottsWebGPU({ baseUrl: '/weights', precision: 'fp16' });
const out = await model.run(model.structureFromPDB(pdbText));
console.log(out.timings.gpuMs);                 // encoder + head, submit to completion
model.destroy();                                // frees every GPU buffer
```

Both handles expose the same analysis surface, so code written against one runs on
the other:

```js
const centres = model.resolveCentres(structure, [
  { chain: 'A', resSeq: 42, protonationType: 'HIS-P' },
  { position: 61,          protonationType: 'GLU-P' },
]);

model.selectiveGap(out, centres);               // { centres: [{ gap }], gapSum }

const designable = model.centreNeighbourhood(out, centres, { perCentre: 8, max: 12 });
const design = model.optimize(out, {
  centres, designable,
  lambda: 0.5,            // 0 = pure stability, 1 = pure selectivity
  blockSize: 2,           // exact argmin over V**2 joint assignments per block
  zscaleMode: 'block',
  temperature: 0,         // > 0 samples the block instead
});
design.sequence           // one-letter canonical sequence
design.tokenNames         // ['ALA', 'HIS-P', 'GLU-D', …]
design.selectiveEnergy    // Σ centres (E_P − E_D)
```

## The objective

The Hamiltonian is the reference's, double-counting convention included, so every
number here is directly comparable to it:

```
E_global = H(s) = Σ_i Σ_k J[i, k, s_i, s_{E_idx[i,k]}]
```

Slot `k = 0` is the self edge (`E_idx[i,0] == i`), forced diagonal, and carries the
single-body field `h_i`; slots `k ≥ 1` carry the pair couplings. The kNN graph is
asymmetric — about one edge in six has no reciprocal partner — so the per-position
conditional energy adds both the outgoing couplings `i → j` *and* the incoming ones
`m → i` (the graph transpose). That is what makes the identity

```
H(s with s_p := a) − H(s) === E[p, a] − E[p, s_p]
```

hold, and the test asserts it directly.

The pH-selective gap pins each centre protonated and contrasts it against its
deprotonated partner, measured in the presence of the other centres:

```
E_P − E_D = Σ_centres ( e_P(centre) − mean_d e_D(centre, d) )      lower = more selective
```

Block descent then minimises `(1 − λ)·z(E_global) + λ·z(E_P − E_D)`. The `z(·)` is
not decoration: the two terms differ by three orders of magnitude, so each is divided
by the spread of its own deltas — frozen once, before the first sweep — which is what
turns `λ` into a true relative weight. Only stability has within-block pairwise terms
(selectivity is exactly unary), so `blockSize` changes the answer solely through
stability coupling.

## Precisions

`tools/export_weights.py` writes three bundles plus a `manifest.json` with SHA-256
digests. 2-D weights are quantized; 1-D tensors (biases, LayerNorm) stay FP32.

| Bundle | On device | What it is | `J` relRMS vs FP32 oracle | GPU vs CPU at the same precision |
| --- | --- | --- | --- | --- |
| `fp32` | 7.12 MB | reference | `1.6e-7` | `1.7e-7` |
| `fp16` | 3.58 MB | IEEE binary16, two per `u32`, `unpack2x16float` in-shader | `1.1e-4` | `1.8e-7` |
| `int8` | 1.85 MB | per-output-row symmetric int8 + FP32 scale, four per `u32` | `6.2e-3` | `1.7e-7` |

Accumulation is always FP32 (the CPU path uses double accumulators, a superset).
The last column is the one that tests the kernels: it compares GPU and CPU *at the
same precision*, so quantization error cancels and only a kernel bug shows up.

`shader-f16` is never required — the FP16 bundle is unpacked to `f32` on read, so it
runs on any adapter.

## Verification

`tools/dump_oracle.py` runs the **real** `mpnn.model.pottsmpnn.PottsMPNN` from the
reference tree in CPU FP32 and records every intermediate: featurizer inputs, the
`edgeIdx` graph, raw and embedded edge features, encoder states, `h`, `J`, Potts
energies, the selective gap, decoder log-probabilities, and block-descent
trajectories. The descent fixtures are produced by the reference `_block_descent`
itself — the oracle drives it through a shim context rather than re-implementing it,
so the JS optimizer is checked against the real algorithm, not a paraphrase.

Two cases: a 24-residue crop straddling the A/B chain boundary (full tensors, and it
exercises the cross-chain positional slot and a token override) and the whole
229-residue two-chain complex (summaries, 4096 sampled `J` entries, full energies).

`test/gpu.test.mjs` drives `test/gpu-page.{html,mjs}` in `chrome-headless-shell` on
the real Vulkan adapter, comparing GPU against both the CPU backend and the oracle.
Override the binary with `CHROME_HEADLESS_SHELL=/path/to/chrome-headless-shell`.

Two things about that harness are worth knowing, because both are easy to get wrong:

* **The page is loaded from `file://`, not from a local HTTP server.** WebGPU needs a
  secure context, and an `about:blank` target created over CDP is not one — its
  `isSecureContext` is false and `navigator.gpu` is not even defined. `file://` is
  trustworthy, so `--allow-file-access-from-files` plus XHR (not `fetch`, which does
  not serve local files) is what loads the modules and the weight bundles.
* **The page is opened with `Target.createTarget` and answers through
  `Runtime.evaluate`.** The headless shell ignores a positional URL once
  `--remote-debugging-port` is set, and the report comes back as the resolved value
  of `window.__protonPottsReport` rather than over the network.

### Measured on an NVIDIA GB10 (Chrome for Testing 153, Vulkan)

Warm GPU time is the encoder + Potts head, submit to `onSubmittedWorkDone`, median of
12 runs on a compiled plan:

| Case | Warm GPU | Cold featurize / upload / readback | Pure-JS CPU backend |
| --- | --- | --- | --- |
| 24-residue crop (`L=24, K=24`) | **3.1 ms** | 4.9 / 0.8 / 8.4 ms | 208 ms |
| full complex (`L=229, K=48`) | **15.3 ms** | 38.2 / 8.0 / 38.1 ms | 3542 ms |

So the kernels are ~230x the pure-JS path on the full complex. Readback dominates the
end-to-end cold number, because `J` is `L·K·V·V` floats — 40 MB at `L=229`; an
application that keeps the optimizer on the GPU would not pay it.

The teacher-forced decoder adds 5.7–6.3 ms on the complex.

## Licence

MIT — see `LICENSE`.
