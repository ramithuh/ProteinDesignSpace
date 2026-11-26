import torch


def e_dot_v(g, x, y):
    _, dst = g.edges()
    return (x * y[dst]).sum(dim=-1, keepdim=True)


def edge_softmax(g, logits, eids=None, norm_by="dst"):
    _, dst = g.edges()
    n = g.num_nodes()
    shape = (n,) + tuple(logits.shape[1:])
    mx = torch.full(shape, float("-inf"), dtype=logits.dtype)
    mx = mx.index_reduce_(0, dst, logits, "amax", include_self=True)
    ex = torch.exp(logits - mx[dst])
    den = torch.zeros(shape, dtype=logits.dtype).index_add_(0, dst, ex)
    return ex / den[dst]


def copy_e_sum(g, x):
    _, dst = g.edges()
    out = torch.zeros((g.num_nodes(),) + tuple(x.shape[1:]), dtype=x.dtype)
    return out.index_add_(0, dst, x)
