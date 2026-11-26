#!/usr/bin/env python
"""Export a cg2all PyTorch Lightning checkpoint to WebGPU weight bundles.

Writes <out>/fp16/{manifest.json,weights.bin} and <out>/int8/{manifest.json,weights.bin}
(add --precisions fp32,fp16,int8 for an unquantized fp32 reference bundle).
Every state_dict tensor is kept under its PyTorch name (without the Lightning
"model." prefix) so the JS loader maps 1:1 onto libmodel.py / SE3Transformer.

Only the radial-profile output layers (`*.radial_func.net.6.weight`, 3.9 M of the
4.05 M parameters) are stored in reduced precision; every other tensor (embedding,
LinearSE3 maps, norms, radial MLP hidden layers) stays float32 (~0.6 MiB). The
LinearSE3 maps are the precision-sensitive part: rounding them to fp16 alone moves
side-chain hydrogens by up to 1.7 A, while fp16 radial weights move atoms < 0.01 A.

fp16: radial output layers as IEEE half.
int8: radial output layers as symmetric int8 with one fp16 scale per 16
      consecutive hidden inputs (two scales per (c_out, c_in, frequency) radial
      weight row): +0.23 MiB over per-row scales, about half the backmapping error.
fp32: (optional, --precisions) unquantized reference bundle.

usage:
  python -I export_weights.py --ckpt CalphaBasedModel.ckpt --variant ca --out /tmp/cg2all-weights/ca
Only torch and numpy are needed (the checkpoint is a pickle: load trusted files only).
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

REDUCED_SUFFIX = "radial_func.net.6.weight"
INT8_GROUP = 16
ALIGN = 16

VARIANTS = {
    "ca": {"cg_model": "CalphaBasedModel", "file": "CalphaBasedModel.ckpt", "fix_atom": False},
    "ca-fix": {"cg_model": "CalphaBasedModel", "file": "CalphaBasedModel-FIX.ckpt", "fix_atom": True},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(obj):
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def quantize_int8(w):
    rows = w.reshape(-1, w.shape[-1])
    group = INT8_GROUP if rows.shape[1] % INT8_GROUP == 0 else rows.shape[1]
    g = rows.reshape(rows.shape[0], -1, group)
    scale = (np.abs(g).max(axis=-1) / 127.0).astype(np.float16)
    scale[scale == 0] = np.float16(1.0)
    q = np.clip(np.rint(g / scale.astype(np.float32)[..., None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale.reshape(-1), group


def write_bundle(state, out_dir, precision, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    tensors = []

    def push(arr):
        while len(blob) % ALIGN:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        return off, arr.nbytes

    max_err = 0.0
    for name, t in state.items():
        w = t.detach().cpu().numpy().astype(np.float32)
        entry = {"name": name, "shape": list(w.shape)}
        reduced = name.endswith(REDUCED_SUFFIX)
        if precision == "fp32" or not reduced:
            off, nbytes = push(w)
            entry.update(dtype="float32", offset=off, bytes=nbytes)
        elif precision == "int8":
            q, scale, group = quantize_int8(w)
            off, nbytes = push(q)
            soff, snbytes = push(scale)
            entry.update(dtype="int8", offset=off, bytes=nbytes,
                         scale={"dtype": "float16", "offset": soff, "bytes": snbytes, "groupSize": group})
            deq = (q.reshape(-1, group).astype(np.float32) * scale.astype(np.float32)[:, None]).reshape(w.shape)
            max_err = max(max_err, float(np.abs(deq - w).max()))
        else:
            h = w.astype(np.float16)
            off, nbytes = push(h)
            entry.update(dtype="float16", offset=off, bytes=nbytes)
            max_err = max(max_err, float(np.abs(h.astype(np.float32) - w).max()))
        tensors.append(entry)

    bin_path = out_dir / "weights.bin"
    bin_path.write_bytes(bytes(blob))
    manifest = {
        "format": "cg2all-webgpu/v1",
        "precision": precision,
        **meta,
        "files": [{"name": "weights.bin", "bytes": len(blob), "sha256": sha256(bin_path)}],
        "maxAbsQuantError": max_err,
        "tensors": tensors,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--precisions", default="fp16,int8")
    args = ap.parse_args()

    info = VARIANTS[args.variant]
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = plain(ckpt["hyper_parameters"])
    if hp["cg_model"] != info["cg_model"]:
        raise SystemExit(f"checkpoint cg_model {hp['cg_model']} != {info['cg_model']}")
    if bool(hp["globals"].get("fix_atom", False)) != info["fix_atom"]:
        raise SystemExit("checkpoint fix_atom flag does not match the requested variant")
    state = {".".join(k.split(".")[1:]): v for k, v in ckpt["state_dict"].items()}

    meta = {
        "variant": args.variant,
        "cgModel": hp["cg_model"],
        "fixAtom": info["fix_atom"],
        "config": {"globals": hp["globals"], "embedding_module": hp["embedding_module"],
                   "structure_module": {k: v for k, v in hp["structure_module"].items() if k != "loss_weight"}},
        "source": {
            "checkpoint": info["file"],
            "url": f"https://zenodo.org/record/8393343/files/{info['file']}",
            "sha256": sha256(args.ckpt),
            "epoch": ckpt.get("epoch"),
            "license": "Apache-2.0",
            "upstream": "https://github.com/huhlim/cg2all",
        },
        "numParams": int(sum(v.numel() for v in state.values())),
    }
    out = pathlib.Path(args.out)
    for precision in args.precisions.split(","):
        m = write_bundle(state, out / precision, precision, meta)
        print(f"{args.variant}/{precision}: {m['files'][0]['bytes'] / 2**20:.2f} MiB "
              f"sha256={m['files'][0]['sha256'][:16]}.. maxAbsErr={m['maxAbsQuantError']:.3g}")


if __name__ == "__main__":
    main()
