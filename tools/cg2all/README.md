# cg2all port tooling

Offline scripts behind the `cg2all` plugin (`web/src/lib/plugins/cg2all`, runtime in
`web/static/vendor/cg2all`). None of this runs in the browser.

| Script | Produces |
| --- | --- |
| `export_weights.py` | `fp16` / `int8` (optionally `fp32`) weight bundles — `manifest.json` (tensor table, config, SHA-256) + `weights.bin` — for `ramithuh/cg2all-webgpu/<variant>/<precision>/` |
| `export_constants.py` | `web/static/vendor/cg2all/shared/constants.js` (CHARMM36 residue topology, rigid groups and transforms per SS class, e3nn Wigner-3j, MDTraj PDB names) |
| `make_parity_inputs.py` | the three parity inputs (16-aa ideal helix, crambin 1AB1 Cα trace with SSBONDs, two-chain trace with a 14.6 Å break + Zn) |
| `reference_forward.py` | PyTorch goldens from the unmodified cg2all + SE3Transformer code (`shim/dgl` stands in for DGL) |
| `build_test_fixture.py` | `web/src/lib/__tests__/fixtures/cg2all-parity.json` from the goldens |
| `webgpu-parity.mjs` | all-atom RMSD of the WGSL engine vs the goldens in headless Chromium (SwiftShader WebGPU) |

```bash
# Python: torch (CPU), numpy, e3nn==0.5.1, ml-collections, mdtraj; checkouts of huhlim/cg2all and huhlim/SE3Transformer
curl -LO https://zenodo.org/record/8393343/files/CalphaBasedModel.ckpt
curl -LO https://zenodo.org/record/8393343/files/CalphaBasedModel-FIX.ckpt
python -I tools/cg2all/export_weights.py --ckpt CalphaBasedModel.ckpt --variant ca --out /tmp/cg2all-weights/ca
python -I tools/cg2all/export_weights.py --ckpt CalphaBasedModel-FIX.ckpt --variant ca-fix --out /tmp/cg2all-weights/ca-fix
python -I tools/cg2all/export_constants.py --cg2all ../cg2all --out web/static/vendor/cg2all/shared/constants.js

python -I tools/cg2all/make_parity_inputs.py /tmp/cg2all-inputs ../cg2all/tests
for ck in CalphaBasedModel CalphaBasedModel-FIX; do for f in helix16 crambin46 twochain; do
  python -I tools/cg2all/reference_forward.py --cg2all ../cg2all --se3 ../SE3Transformer \
    --ckpt $ck.ckpt --pdb /tmp/cg2all-inputs/$f.pdb --out /tmp/cg2all-golden/$f.$ck.json
done; done
python -I tools/cg2all/build_test_fixture.py --inputs /tmp/cg2all-inputs --golden /tmp/cg2all-golden \
  --manifest /tmp/cg2all-weights/ca/fp16/manifest.json --out web/src/lib/__tests__/fixtures/cg2all-parity.json

# WebGPU parity (any Chromium with WebGPU; SwiftShader works headless)
EXE=/path/to/chrome CASES=ca:CalphaBasedModel:fp16,ca-fix:CalphaBasedModel-FIX:int8 node tools/cg2all/webgpu-parity.mjs
# Full-network CPU parity inside vitest runs whenever CG2ALL_WEIGHTS_DIR (default /tmp/cg2all-weights) holds the bundles
cd web && npx vitest run src/lib/__tests__/cg2all.test.ts
```

Notes on fidelity:

- The final `ConvSE3` of the SE(3)-Transformer is built with the default fuse level, so it runs the
  per-output-degree fused path whose (0,0) basis block is the constant Y⁰ = 1/√(4π); the port folds that
  constant into the radial weights. The attention-block convolutions run pairwise (`low_memory=True`).
- `libcg.get_geometry` builds dihedral quadruplets with `torch.stack([...]).reshape(-1, 4, 3)` (rows of the
  stacked tensor, not consecutive residues) and leaves the last residue's "next" bond lengths at 0; the
  checkpoints were trained that way and `shared/features.js` reproduces both.
- Upstream picks HSD/HSE at random for HIS; the port (and `reference_forward.py`) always uses HSD.
