"""Turn one OPFData scenario dict into a model-ready instance.

Bus, branch and generator tables go in; an instance, its TaskSpec and the solver
labels come out, including the generator/bus allocation the cost depends on.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from powerform.data.cost_features import (
    cost_feats_from_opfdata, ed_prior_from_opfdata, global_context_features,
)
from powerform.model.spec import TaskSpec
from powerform.physics.newton_solver import TransmissionGrid

BUS_TYPE_REF = 3

# nodes.bus
B_TYPE, B_VMIN, B_VMAX = 1, 2, 3
# nodes.generator
G_PMIN, G_PMAX, G_QMIN, G_QMAX, G_C2, G_C1, G_C0 = 2, 3, 5, 6, 8, 9, 10
# nodes.load / nodes.shunt
L_PD, L_QD = 0, 1
S_BS, S_GS = 0, 1
# edges
AC_BFR, AC_BTO, AC_R, AC_X, AC_RATE = 2, 3, 4, 5, 6
TR_R, TR_X, TR_RATE, TR_TAP, TR_SHIFT, TR_BFR, TR_BTO = 2, 3, 4, 7, 8, 9, 10
# solution.bus / solution.generator
SOL_THETA, SOL_VM = 0, 1
SOL_PG, SOL_QG = 0, 1

# Branches whose rate_a is 0 are unlimited in pglib.
UNRATED_SENTINEL = 99.0

# Widths of inst.cost_feats a checkpoint can ask for: 12 with the
# economic-dispatch prior, 10 without; at 0 the model reads none.
N_COST_FEATS_LAYOUTS = (0, 10, 12)


class OPFDataLabels:
    """Solver labels for one scenario, per generator and per bus.

    vm, va:            (N,) label voltage magnitude and angle (rad)
    pg_gen, qg_gen:    (G,) per-generator dispatch (pu)
    gen_bus:           (G,) bus index of each generator
    c2, c1, c0:        (G,) cost coefficients on the pu-Pg basis
    pmin_gen, pmax_gen, qmin_gen, qmax_gen: (G,) per-generator limits (pu)
    pg_bus, qg_bus:    (N,) bus-aggregated dispatch (pu)
    cost:              sum_g (c2 pg^2 + c1 pg + c0) at the label dispatch
    objective:         ``metadata.objective`` as shipped, for cross-check
    """

    def __init__(self, obj: Dict[str, Any], device) -> None:
        sol_bus = np.asarray(obj["solution"]["nodes"]["bus"], dtype=float)
        sol_gen = np.asarray(obj["solution"]["nodes"]["generator"], dtype=float)
        gx = np.asarray(obj["grid"]["nodes"]["generator"], dtype=float)
        gen_bus = np.asarray(obj["grid"]["edges"]["generator_link"]["receivers"],
                             dtype=np.int64)
        n = len(obj["grid"]["nodes"]["bus"])

        t = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
        self.vm = t(sol_bus[:, SOL_VM])
        self.va = t(sol_bus[:, SOL_THETA])
        self.pg_gen = t(sol_gen[:, SOL_PG])
        self.qg_gen = t(sol_gen[:, SOL_QG])
        self.gen_bus = torch.tensor(gen_bus, dtype=torch.int64, device=device)
        self.c2, self.c1, self.c0 = t(gx[:, G_C2]), t(gx[:, G_C1]), t(gx[:, G_C0])
        self.pmin_gen, self.pmax_gen = t(gx[:, G_PMIN]), t(gx[:, G_PMAX])
        self.qmin_gen, self.qmax_gen = t(gx[:, G_QMIN]), t(gx[:, G_QMAX])

        self.pg_bus = torch.zeros(n, dtype=torch.float64, device=device)
        self.qg_bus = torch.zeros(n, dtype=torch.float64, device=device)
        self.pg_bus.index_add_(0, self.gen_bus, self.pg_gen)
        self.qg_bus.index_add_(0, self.gen_bus, self.qg_gen)

        self.cost = float(gen_cost(self.pg_gen, self.c2, self.c1, self.c0))
        self.objective = float(obj["metadata"]["objective"])


def gen_cost(pg: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor,
             c0: torch.Tensor) -> torch.Tensor:
    """sum_g (c2 pg^2 + c1 pg + c0)."""
    return (c2 * pg.pow(2) + c1 * pg + c0).sum()


def _scatter_sum(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    np.add.at(out, index, values)
    return out


class _Inst:
    """Attribute bag for one instance."""


def build_instance(obj: Dict[str, Any], device, n_cost_feats: int):
    """OPFData scenario dict -> (inst, spec, labels).

    ``n_cost_feats`` selects the width of ``inst.cost_feats`` and must match the
    checkpoint; a mismatch is a shape error in ``embed_s_var``. See
    ``data/cost_features.py`` for the columns.
    """
    if n_cost_feats not in N_COST_FEATS_LAYOUTS:
        raise ValueError(f"n_cost_feats={n_cost_feats} is not one of "
                         f"{N_COST_FEATS_LAYOUTS}")

    nodes, edges = obj["grid"]["nodes"], obj["grid"]["edges"]
    bus = np.asarray(nodes["bus"], dtype=float)
    n = bus.shape[0]

    is_ref = bus[:, B_TYPE] == BUS_TYPE_REF
    if not is_ref.any():
        raise ValueError("scenario has no reference bus")

    # loads and shunts are many-to-one onto buses
    load = np.asarray(nodes["load"], dtype=float).reshape(-1, 2)
    load_bus = np.asarray(edges["load_link"]["receivers"], dtype=np.int64)
    p_load = _scatter_sum(load[:, L_PD], load_bus, n)
    q_load = _scatter_sum(load[:, L_QD], load_bus, n)

    shunt = np.asarray(nodes["shunt"], dtype=float).reshape(-1, 2)
    shunt_bus = np.asarray(edges["shunt_link"]["receivers"], dtype=np.int64)
    bs = _scatter_sum(shunt[:, S_BS], shunt_bus, n)
    gs = _scatter_sum(shunt[:, S_GS], shunt_bus, n)

    # ac_line and transformer into one MATPOWER-style branch table
    ac_f = np.asarray(edges["ac_line"]["features"], dtype=float).reshape(-1, 9)
    ac_s = np.asarray(edges["ac_line"]["senders"], dtype=np.int64)
    ac_r = np.asarray(edges["ac_line"]["receivers"], dtype=np.int64)
    tr_f = np.asarray(edges["transformer"]["features"],
                      dtype=float).reshape(-1, 11)
    tr_s = np.asarray(edges["transformer"]["senders"], dtype=np.int64)
    tr_r = np.asarray(edges["transformer"]["receivers"], dtype=np.int64)

    # The pi-model splits br_b symmetrically, which can only represent
    # b_fr == b_to. Refuse a real asymmetry rather than average it away.
    for name, bfr, bto in (("ac_line", ac_f[:, AC_BFR], ac_f[:, AC_BTO]),
                           ("transformer", tr_f[:, TR_BFR], tr_f[:, TR_BTO])):
        if bfr.size and not np.allclose(bfr, bto, rtol=0, atol=1e-12):
            worst = int(np.argmax(np.abs(bfr - bto)))
            raise ValueError(
                f"{name} charging susceptance is asymmetric "
                f"(b_fr={bfr[worst]!r} != b_to={bto[worst]!r} at edge {worst}); "
                f"the symmetric pi-model cannot represent it")

    f_bus = np.concatenate([ac_s, tr_s])
    t_bus = np.concatenate([ac_r, tr_r])
    br_r = np.concatenate([ac_f[:, AC_R], tr_f[:, TR_R]])
    br_x = np.concatenate([ac_f[:, AC_X], tr_f[:, TR_X]])
    br_b = np.concatenate([ac_f[:, AC_BFR] + ac_f[:, AC_BTO],
                           tr_f[:, TR_BFR] + tr_f[:, TR_BTO]])
    tap = np.concatenate([np.ones(len(ac_s)), tr_f[:, TR_TAP]])
    tap = np.where(tap == 0.0, 1.0, tap)
    shift = np.concatenate([np.zeros(len(ac_s)), tr_f[:, TR_SHIFT]])
    rate_raw = np.concatenate([ac_f[:, AC_RATE], tr_f[:, TR_RATE]])
    # rate_raw keeps the unrated zeros: the graph context averages the ratings
    # as shipped, and the sentinel would inflate log1p(mean rate) on every grid
    # with unrated branches.
    rate = np.where(rate_raw <= 0, UNRATED_SENTINEL, rate_raw)

    labels = OPFDataLabels(obj, device)

    d = device
    tt = lambda a, dt=torch.float64: torch.tensor(a, dtype=dt, device=d)
    slack = tt(is_ref, torch.bool)

    tgrid = TransmissionGrid(
        f_bus=tt(f_bus, torch.int64), t_bus=tt(t_bus, torch.int64),
        br_r=tt(br_r), br_x=tt(br_x), br_b=tt(br_b),
        tap=tt(tap), shift_rad=tt(shift),
        gs=tt(gs), bs=tt(bs),
        p_load=tt(p_load), q_load=tt(q_load),
        is_slack=slack,
    )

    # generator slots, aggregated per bus (the dispatch granularity)
    gx = np.asarray(nodes["generator"], dtype=float)
    gen_bus = np.asarray(edges["generator_link"]["receivers"], dtype=np.int64)
    p_min = _scatter_sum(gx[:, G_PMIN], gen_bus, n)
    p_max = _scatter_sum(gx[:, G_PMAX], gen_bus, n)
    has_gen = np.zeros(n, dtype=bool)
    has_gen[gen_bus] = True
    m_node_ns = tt(has_gen, torch.bool) & ~slack       # non-slack gen buses
    m_free = m_node_ns | slack

    # The generation box the model decides over, zeroed at the reference bus
    # whose active power the power flow sets.
    zero_n = torch.zeros(n, dtype=torch.float64, device=d)
    spec_p_min = torch.where(slack, zero_n, tt(p_min))
    spec_p_max = torch.where(slack, zero_n, tt(p_max))

    v_slack = (labels.vm * torch.exp(1j * labels.va)).to(torch.complex128)
    v_slack = torch.where(slack, v_slack, torch.ones_like(v_slack))

    inst = _Inst()
    inst.tgrid = tgrid
    inst.edge_uv = torch.stack([tgrid.f_bus, tgrid.t_bus], dim=1)
    inst.edge_r = tgrid.br_r.clamp_min(1e-6)
    inst.edge_x = tgrid.br_x.abs().clamp_min(1e-6)
    inst.s_max = tt(rate)
    inst.p_load = tgrid.p_load
    inst.q_load = tgrid.q_load
    inst.is_sub = tgrid.is_slack
    inst.n_nodes = n
    inst.n_edges = int(inst.edge_uv.shape[0])
    inst.pg_label = labels.pg_bus
    inst.vm_ref = labels.vm
    inst.va_ref = labels.va
    inst.v_slack_scalar = v_slack.abs()
    inst.depth = _bfs_depth(f_bus, t_bus, is_ref, n).to(d)
    inst.cost_feats = cost_feats_from_opfdata(labels, n, slack, m_node_ns)
    if n_cost_feats in (10, 12):
        # The generator totals cover all units including those on the reference
        # bus, and the ratings are the raw values with unrated branches at zero.
        inst.cost_feats = torch.cat([
            inst.cost_feats,
            global_context_features(
                p_load_pu=tgrid.p_load, q_load_pu=tgrid.q_load,
                gen_pmax_pu=tt(gx[:, G_PMAX]),
                gen_qmax_pu=tt(gx[:, G_QMAX]),
                gen_qmin_pu=tt(gx[:, G_QMIN]),
                vm_min=tt(bus[:, B_VMIN]), vm_max=tt(bus[:, B_VMAX]),
                rate_pu=tt(rate_raw), br_r=tgrid.br_r, br_x=tgrid.br_x,
                n_bus=n),
        ], dim=-1)
    if n_cost_feats == 12:
        inst.cost_feats = torch.cat([
            inst.cost_feats,
            ed_prior_from_opfdata(
                labels, n, slack, m_node_ns,
                p_load_total=float(tgrid.p_load.sum()),
                p_min=spec_p_min, p_max=spec_p_max),
        ], dim=-1)
    inst.m_node_ns = m_node_ns
    inst.m_free = m_free

    spec = TaskSpec(
        m_node=m_free,
        m_edge=torch.zeros(inst.n_edges, dtype=torch.bool, device=d),
        p_min=spec_p_min, p_max=spec_p_max,
        vm_min=tt(bus[:, B_VMIN]), vm_max=tt(bus[:, B_VMAX]))
    return inst, spec, labels


def _bfs_depth(f_bus: np.ndarray, t_bus: np.ndarray, is_ref: np.ndarray,
               n: int) -> torch.Tensor:
    """Normalized hop distance from the reference bus.

    Buses unreachable from the reference, possible under a branch removal, get
    the maximum depth.
    """
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in zip(f_bus.tolist(), t_bus.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    dep = np.full(n, -1, dtype=np.int64)
    queue = [i for i in range(n) if is_ref[i]]
    dep[queue] = 0
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        for j in adj[i]:
            if dep[j] < 0:
                dep[j] = dep[i] + 1
                queue.append(j)
    depf = np.where(dep < 0, float(n), dep).astype(float)
    return torch.tensor(depf / max(depf.max(), 1.0), dtype=torch.float64)


def allocate_bus_p_to_gens(p_bus: torch.Tensor, labels: OPFDataLabels,
                           n_bisect: int = 60) -> torch.Tensor:
    """Split a per-bus active dispatch across that bus's generators.

    The decision variable is the bus-aggregated Pg, while cost is per generator.
    The split is equal-incremental-cost dispatch, what an OPF would do with the
    same bus total:

        p_g(lambda) = clip((lambda - c1_g) / (2 c2_g), pmin_g, pmax_g)

    solved for the lambda that meets the bus total. Linear-cost units (c2 == 0)
    switch at lambda == c1 and are handled by the same clip through a large
    slope. Sixty halvings take the bracket below float64 resolution.
    """
    gen_bus = labels.gen_bus
    c2, c1 = labels.c2, labels.c1
    pmin, pmax = labels.pmin_gen, labels.pmax_gen
    slope = 1.0 / (2.0 * c2).clamp_min(1e-9)

    def dispatch_at(lam_bus: torch.Tensor):
        p = ((lam_bus[gen_bus] - c1) * slope).clamp(min=pmin, max=pmax)
        total = torch.zeros_like(p_bus).index_add_(0, gen_bus, p)
        return total, p

    lo = torch.full_like(p_bus, -1e6)
    hi = torch.full_like(p_bus, 1e6)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        total, _ = dispatch_at(mid)
        below = total < p_bus
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    _total, p_gen = dispatch_at(0.5 * (lo + hi))
    return p_gen


def allocate_bus_q_to_gens(q_bus: torch.Tensor,
                           labels: OPFDataLabels) -> torch.Tensor:
    """Split a per-bus reactive dispatch across that bus's generators.

    Proportional sharing on the reactive range, as power-flow programs do when
    several units regulate one bus:

        q_g = qmin_g + (Q_bus - sum qmin) * (qmax_g - qmin_g) / sum(range)

    Buses whose units have zero total range use an equal split.
    """
    gen_bus = labels.gen_bus
    qmin, qmax = labels.qmin_gen, labels.qmax_gen
    rng = (qmax - qmin).clamp_min(0.0)
    n = q_bus.shape[0]
    rng_bus = torch.zeros(n, dtype=rng.dtype,
                          device=rng.device).index_add_(0, gen_bus, rng)
    qmin_bus = torch.zeros(n, dtype=qmin.dtype,
                           device=qmin.device).index_add_(0, gen_bus, qmin)
    cnt_bus = torch.zeros(n, dtype=rng.dtype, device=rng.device).index_add_(
        0, gen_bus, torch.ones_like(rng))

    share = torch.where(rng_bus[gen_bus] > 1e-9,
                        rng / rng_bus[gen_bus].clamp_min(1e-9),
                        1.0 / cnt_bus[gen_bus].clamp_min(1.0))
    return qmin + (q_bus - qmin_bus)[gen_bus] * share
