"""Minimal pure-PyTorch stand-in for the DGL APIs used by cg2all inference.

Only implements what cg2all/SE3Transformer touch at inference time:
radius_graph, graph(), DGLGraph.{edges,ndata,edata,in_degrees,nodes,
has_edges_between,edge_ids,num_nodes,num_edges,to}, ops.{e_dot_v,edge_softmax,
copy_e_sum}. Semantics follow DGL 1.1.0 (dst-normalised softmax, e_dot_v uses
the destination node, copy_e_sum reduces onto dst).
"""
import torch
from . import ops  # noqa: F401
from . import nn  # noqa: F401
from . import dataloading  # noqa: F401


class DGLGraph:
    def __init__(self, src, dst, num_nodes):
        self._src = src.long()
        self._dst = dst.long()
        self._n = int(num_nodes)
        self.ndata = {}
        self.edata = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._n

    def num_edges(self):
        return int(self._src.shape[0])

    def nodes(self):
        return torch.arange(self._n)

    def in_degrees(self, v=None):
        deg = torch.bincount(self._dst, minlength=self._n)
        return deg if v is None else deg[v]

    def has_edges_between(self, u, v):
        u = torch.as_tensor(u).long().reshape(-1)
        v = torch.as_tensor(v).long().reshape(-1)
        key = self._src * self._n + self._dst
        q = u * self._n + v
        return torch.isin(q, key)

    def edge_ids(self, u, v):
        u = torch.as_tensor(u).long().reshape(-1)
        v = torch.as_tensor(v).long().reshape(-1)
        key = (self._src * self._n + self._dst).tolist()
        lut = {k: i for i, k in enumerate(key)}
        return torch.as_tensor([lut[int(a) * self._n + int(b)] for a, b in zip(u, v)], dtype=torch.long)

    def to(self, device):
        return self


def graph(edges, num_nodes=None, device=None):
    src, dst = edges
    src = torch.as_tensor(src)
    dst = torch.as_tensor(dst)
    if num_nodes is None:
        num_nodes = int(max(src.max(), dst.max()) + 1) if src.numel() else 0
    return DGLGraph(src, dst, num_nodes)


def radius_graph(x, r, p=2, self_loop=False, compute_mode="donot_use_mm_for_euclid_dist", get_distances=False):
    distances = torch.cdist(x, x, p=p, compute_mode=compute_mode)
    if not self_loop:
        distances.fill_diagonal_(r + 1)
    edges = torch.nonzero(distances <= r, as_tuple=True)
    return graph(edges, num_nodes=x.shape[0])


def unbatch(g):
    return [g]
