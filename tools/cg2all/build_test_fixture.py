#!/usr/bin/env python
"""Pack reference_forward.py goldens into web/src/lib/__tests__/fixtures/cg2all-parity.json.

usage: python -I build_test_fixture.py --inputs <dir> --golden <dir> --manifest <ca/fp16/manifest.json> --out <json>
Goldens are expected as <golden>/<name>.CalphaBasedModel.json and <name>.CalphaBasedModel-FIX.json.
"""
import argparse
import json

import numpy as np
import torch
import e3nn.o3 as o3


def rnd(x, sig):
    return [float(f"{v:.{sig}g}") for v in np.asarray(x, dtype=float).ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {
        "source": "huhlim/cg2all a00b881 CalphaBasedModel(-FIX) PyTorch FP32 forward (tools/cg2all/reference_forward.py)",
        "cases": {},
    }
    for name in ["helix16", "crambin46", "twochain"]:
        g = json.load(open(f"{args.golden}/{name}.CalphaBasedModel.json"))
        case = {"pdb": open(f"{args.inputs}/{name}.pdb").read(), "n": g["n"], "e": g["e"]}
        if name != "twochain":
            case["src"] = [int(v) for v in g["src"]]
            case["dst"] = [int(v) for v in g["dst"]]
            ef = np.asarray(g["edge_feat_0"])
            case["edgeClass"] = "".join(
                "x" if r[0] and r[1] else "b" if r[0] else "s" if r[1] else "." for r in ef
            )
            case["f0"] = rnd(g["node_feat_0"], 6)
            case["f1"] = rnd(g["node_feat_1"], 6)
        case["residueType"] = [int(v) for v in g["residue_type"]]
        case["mask"] = "".join(str(int(v)) for v in np.asarray(g["output_atom_mask"]).ravel())
        for variant, ckpt in [("ca", "CalphaBasedModel"), ("ca-fix", "CalphaBasedModel-FIX")]:
            gv = json.load(open(f"{args.golden}/{name}.{ckpt}.json"))
            case[variant] = {
                "s": rnd(gv["struct"]["0"], 6),
                "v": rnd(gv["struct"]["1"], 6),
                "ss": [int(x) for x in gv["ss"]],
                "R": rnd(gv["R"], 5),
            }
        out["cases"][name] = case
    manifest = json.load(open(args.manifest))
    out["tensorShapes"] = [[t["name"], t["shape"]] for t in manifest["tensors"]]
    vec = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float64)
    sh = o3.spherical_harmonics(list(range(5)), vec, normalize=True).numpy()
    out["shProbe"] = {"vec": [0.3, -0.5, 0.8], "sh": rnd(sh, 12)}
    json.dump(out, open(args.out, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
