#!/usr/bin/env python
"""Golden reference for the cg2all WebGPU port: run the unmodified cg2all CalphaBasedModel.

Imports huhlim/cg2all and huhlim/SE3Transformer as-is; only DGL is replaced by the small
pure-PyTorch shim in ./shim (radius_graph, e_dot_v, dst edge_softmax, copy_e_sum with DGL 1.1
semantics) because DGL has no wheels for current torch on aarch64. Dumps graph/features, every
SE(3)-Transformer layer output, the structure-module output and the 24-atom coordinates (nm).

usage:
  python -I reference_forward.py --cg2all ~/src/cg2all --se3 ~/src/SE3Transformer \
      --ckpt CalphaBasedModel.ckpt --pdb in.pdb --out golden.json [--out-pdb out.pdb]
Needs torch, numpy, e3nn==0.5.1, ml-collections, mdtraj. HIS is renamed to HSD first
(cg2all otherwise picks HSD/HSE at random; the port always uses HSD).
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cg2all", required=True, help="checkout of github.com/huhlim/cg2all")
    ap.add_argument("--se3", required=True, help="checkout of github.com/huhlim/SE3Transformer")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-pdb")
    args = ap.parse_args()
    shim = pathlib.Path(__file__).resolve().parent / "shim"
    sys.path[:0] = [str(shim), f"{args.cg2all}/cg2all/lib", args.cg2all, args.se3]

    import numpy as np
    import torch
    import mdtraj

    torch.set_num_threads(4)
    load = mdtraj.load

    def load_with_bfactors(*a, **kw):
        t = load(*a, **kw)
        t.bfactors = np.zeros((t.n_frames, t.n_atoms))  # huhlim/mdtraj fork attribute; unused by the model
        return t

    mdtraj.load = load_with_bfactors
    import libcg
    import libmodel
    from libdata import PredictionData, create_trajectory_from_batch, standardize_atom_name
    from libter import patch_termini

    txt = open(args.pdb).read().replace(" HIS ", " HSD ")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    tmp.write(txt)
    tmp.close()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cg_model = libcg.CalphaBasedModel
    config = libmodel.set_model_config(ckpt["hyper_parameters"], cg_model, flattened=False)
    model = libmodel.Model(config, cg_model, compute_loss=False)
    model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in ckpt["state_dict"].items()})
    model.set_constant_tensors("cpu")
    model.eval()

    data = PredictionData(tmp.name, cg_model, radius=config.globals.radius, chain_break_cutoff=1.0,
                          is_all=False, fix_atom=config.globals.fix_atom)
    batch = data[0]

    taps = {}

    def hook(name):
        def f(mod, inp, out):
            if isinstance(out, dict):
                taps[name] = {k: v.detach().clone() for k, v in out.items()}
        return f

    graph_modules = model.interaction_module.graph_module.graph_modules
    for i, m in enumerate(graph_modules):
        m.register_forward_hook(hook(f"graph.{i}"))
    model.initialization_module.register_forward_hook(hook("init"))
    model.structure_module.register_forward_hook(hook("struct"))
    with torch.no_grad():
        ret = model.forward(batch)[0]

    def T(x):
        return np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x).astype(float).round(7).tolist()

    src, dst = batch.edges()
    golden = {
        "fix_atom": bool(config.globals.get("fix_atom", False)),
        "n": int(batch.num_nodes()),
        "e": int(batch.num_edges()),
        "pos": T(batch.ndata["pos"]),
        "residue_type": T(batch.ndata["residue_type"]),
        "continuous": T(batch.ndata["continuous"]),
        "node_feat_0": T(batch.ndata["node_feat_0"][..., 0]),
        "node_feat_1": T(batch.ndata["node_feat_1"]),
        "src": T(src),
        "dst": T(dst),
        "edge_feat_0": T(batch.edata["edge_feat_0"][..., 0]),
        "ssbond_index": T(batch.ndata["ssbond_index"]),
        "output_atom_mask": T(batch.ndata["output_atom_mask"]),
        "init": {k: T(v) for k, v in taps["init"].items()},
        "layers": [{k: T(v) for k, v in taps[f"graph.{i}"].items()} for i in range(len(graph_modules))],
        "struct": {k: T(v) for k, v in taps["struct"].items()},
        "ss": T(ret["ss"]),
        "bb": T(ret["bb"]),
        "sc": T(ret["sc"]),
        "R": T(ret["R"]),
    }
    json.dump(golden, open(args.out, "w"))
    if args.out_pdb:
        traj_s, _ = create_trajectory_from_batch(batch, ret["R"])
        output = patch_termini(traj_s[0])
        standardize_atom_name(output)
        output.save(args.out_pdb)
    os.unlink(tmp.name)
    print(args.pdb, "N", golden["n"], "E", golden["e"])


if __name__ == "__main__":
    main()
