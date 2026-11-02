"""Graph batching: several instances as one disjoint graph.

Every operator in the trunk is local (bus <-> incident line, per-row MLPs and
LayerNorms), so k instances laid side by side with their bus indices offset are
processed as k separate forwards would be, in one call. The one non-local
operation is the per-instance load standardisation, which the merged instance
keeps per instance through ``node_segments``. Outputs match separate forwards
up to float64 rounding: a (kN x d) matmul sums in a different order than a
(N x d) one.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch

from powerform.model.spec import TaskSpec, standardize

_NODE_FIELDS = ("p_load", "q_load", "is_sub", "v_slack_scalar", "depth",
                "cost_feats", "pg_label", "vm_ref", "va_ref", "m_node_ns",
                "m_free")
_EDGE_FIELDS = ("edge_r", "edge_x", "s_max")
_TGRID_EDGE_FIELDS = ("br_x", "br_b", "tap", "shift_rad", "br_r")


def _cat_field(objs, name):
    vals = [getattr(o, name, None) for o in objs]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise ValueError(f"field {name!r} present on some instances only")
    return torch.cat(vals, dim=0)


def merge_instances(insts: Sequence, specs: Sequence[TaskSpec]):
    """Return (inst, spec) for the disjoint union of ``insts``.

    ``inst.node_segments`` and ``inst.edge_segments`` are lists of (start, end)
    row ranges, one per input instance, in input order.
    """
    if len(insts) != len(specs) or not insts:
        raise ValueError("need the same non-empty number of instances and specs")
    node_off, edge_off = [], []
    n_tot = e_tot = 0
    for it in insts:
        node_off.append(n_tot)
        edge_off.append(e_tot)
        n_tot += int(it.n_nodes)
        e_tot += int(it.n_edges)

    m = SimpleNamespace()
    m.n_nodes, m.n_edges = n_tot, e_tot
    m.node_segments = [(o, o + int(it.n_nodes)) for o, it in zip(node_off, insts)]
    m.edge_segments = [(o, o + int(it.n_edges)) for o, it in zip(edge_off, insts)]
    m.edge_uv = torch.cat([it.edge_uv + o for it, o in zip(insts, node_off)], dim=0)
    for name in _NODE_FIELDS + _EDGE_FIELDS:
        v = _cat_field(insts, name)
        if v is not None:
            setattr(m, name, v)
    m.tgrid = SimpleNamespace(**{f: torch.cat([getattr(it.tgrid, f) for it in insts], dim=0)
                                 for f in _TGRID_EDGE_FIELDS})

    spec = TaskSpec(
        m_node=torch.cat([s.m_node for s in specs]),
        m_edge=torch.cat([s.m_edge for s in specs]),
        p_min=torch.cat([s.p_min for s in specs]),
        p_max=torch.cat([s.p_max for s in specs]),
        vm_min=torch.cat([s.vm_min for s in specs]),
        vm_max=torch.cat([s.vm_max for s in specs]),
    )
    return m, spec


def split_nodes(x: torch.Tensor, merged) -> list:
    return [x[a:b] for a, b in merged.node_segments]


def split_edges(x: torch.Tensor, merged) -> list:
    return [x[a:b] for a, b in merged.edge_segments]


def standardize_segments(x: torch.Tensor, segments) -> torch.Tensor:
    """Standardise each segment on its own."""
    return torch.cat([standardize(x[a:b]) for a, b in segments], dim=0)
