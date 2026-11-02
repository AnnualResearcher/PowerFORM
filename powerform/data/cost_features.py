"""Per-bus generation-cost and network-context input features.

The 12-column layout of ``inst.cost_feats`` is: z-scored cost columns [c1', c2'],
the 8-column graph context, and the 2-column economic-dispatch prior.
"""
from __future__ import annotations

import math

import torch

# Guards a bus whose units all declare Pmax = 0 from a 0/0 weighted mean. The
# floor is absolute, so weights must be in MW: per-unit capacities would give a
# zero-Pmax unit a relative weight 100x larger.
WEIGHT_FLOOR = 1e-6      # MW
STD_FLOOR = 1e-6

BASE_MVA = 100.0
N_COST_FEATS = 2   # [c1', c2']


def bus_cost_features(c1_pu: torch.Tensor, c2_pu: torch.Tensor,
                      weight: torch.Tensor, gen_bus: torch.Tensor,
                      n_bus: int, eligible: torch.Tensor) -> torch.Tensor:
    """The ``(n_bus, 2)`` z-scored cost block, carrying merit order rather than
    currency units.

    ``c1_pu``, ``c2_pu``: per-generator coefficients on the per-unit Pg basis.
    ``weight``: per-generator capacity for the weighted mean, in MW.
    ``eligible``: ``(n_bus,)`` bool, the buses the z-score is taken over, namely
    the non-reference generator buses; the others stay zero. A single eligible
    bus has no spread to normalise and also stays zero.
    """
    dtype = torch.float64
    dev = c1_pu.device
    per_bus = bus_cost_means(c1_pu, c2_pu, weight, gen_bus, n_bus)

    feats = torch.zeros(n_bus, N_COST_FEATS, dtype=dtype, device=dev)
    if int(eligible.sum()) > 1:
        for ci, cv in enumerate(per_bus):
            v = cv[eligible]
            feats[eligible, ci] = (v - v.mean()) / v.std().clamp_min(STD_FLOOR)
    return feats


def bus_cost_means(c1_pu: torch.Tensor, c2_pu: torch.Tensor,
                   weight: torch.Tensor, gen_bus: torch.Tensor,
                   n_bus: int):
    """Capacity-weighted per-bus ``(c1, c2)`` before any z-scoring, in original
    units, which is what the economic-dispatch prior prices with."""
    dtype = torch.float64
    dev = c1_pu.device
    w = weight.to(dtype).clamp_min(WEIGHT_FLOOR)
    den = torch.zeros(n_bus, dtype=dtype, device=dev).index_add_(0, gen_bus, w)
    safe = den.clamp_min(WEIGHT_FLOOR)
    out = []
    for c in (c1_pu, c2_pu):
        num = torch.zeros(n_bus, dtype=dtype, device=dev).index_add_(
            0, gen_bus, c.to(dtype) * w)
        out.append(num / safe)
    return out


def ed_prior_from_opfdata(labels, n_bus: int, is_slack: torch.Tensor,
                          eligible: torch.Tensor, p_load_total: float,
                          p_min: torch.Tensor, p_max: torch.Tensor):
    """``economic_dispatch_prior`` for an OPFData scenario. Only units off the
    reference bus contribute, matching ``eligible``."""
    keep = ~is_slack[labels.gen_bus]
    c1_bus, c2_bus = bus_cost_means(
        c1_pu=labels.c1[keep], c2_pu=labels.c2[keep],
        weight=labels.pmax_gen[keep] * BASE_MVA,
        gen_bus=labels.gen_bus[keep], n_bus=n_bus)
    return economic_dispatch_prior(
        p_load_total=p_load_total, c1_bus=c1_bus, c2_bus=c2_bus,
        p_min=p_min, p_max=p_max, eligible=eligible)


def cost_feats_from_opfdata(labels, n_bus: int, is_slack: torch.Tensor,
                            eligible: torch.Tensor) -> torch.Tensor:
    """Cost columns for an OPFData scenario.

    OPFData stores c1 and c2 already on the per-unit Pg basis, so they pass
    through unscaled. Pmax is per-unit and becomes MW for the absolute weight
    floor.
    """
    keep = ~is_slack[labels.gen_bus]
    return bus_cost_features(
        c1_pu=labels.c1[keep], c2_pu=labels.c2[keep],
        weight=labels.pmax_gen[keep] * BASE_MVA,
        gen_bus=labels.gen_bus[keep],
        n_bus=n_bus, eligible=eligible)


def global_context_features(p_load_pu, q_load_pu, gen_pmax_pu, gen_qmax_pu,
                            gen_qmin_pu, vm_min, vm_max, rate_pu,
                            br_r, br_x, n_bus: int) -> torch.Tensor:
    """The 8-column graph context of GridSFM, broadcast per bus:
    [log n_bus, log(n_gen+1), Pd_tot/Pmax_tot, Qd_tot/(Qmax_tot-Qmin_tot),
    mean(Vmax-Vmin), log1p(mean rate), log1p(mean |y|), log1p(max |y|)].

    Tightness gives the operating level, the |y| statistics the transfer
    capability and the logs the scale. Totals are over all generators, slack
    included.
    """
    eps = 1e-6
    n_gen = int(gen_pmax_pu.shape[0])
    pd_tot = float(p_load_pu.sum())
    qd_tot = float(q_load_pu.sum())
    pmax_tot = max(float(gen_pmax_pu.sum()), eps)
    qcap_tot = max(float(gen_qmax_pu.sum()) - float(gen_qmin_pu.sum()), eps)
    v_spread = float((vm_max - vm_min).mean()) if n_bus else 0.0
    z2 = (br_r.double() ** 2 + br_x.double() ** 2).clamp_min(eps)
    abs_y = z2.rsqrt()
    vals = [
        math.log(max(n_bus, 1.0)),
        math.log(n_gen + 1.0),
        pd_tot / pmax_tot,
        qd_tot / qcap_tot,
        v_spread,
        math.log1p(float(rate_pu.mean())),
        math.log1p(float(abs_y.mean())),
        math.log1p(float(abs_y.max())),
    ]
    return torch.tensor(vals, dtype=torch.float64,
                        device=p_load_pu.device).expand(n_bus, 8).clone()


def economic_dispatch_prior(p_load_total: float, c1_bus: torch.Tensor,
                            c2_bus: torch.Tensor, p_min: torch.Tensor,
                            p_max: torch.Tensor, eligible: torch.Tensor,
                            n_iter: int = 60) -> torch.Tensor:
    """Network-free equal-marginal dispatch per bus: the lambda-bisection
    solution of sum_i clip((lam - c1)/(2 c2), Pmin, Pmax) = total load.

    The model then only has to learn the network correction (losses, congestion,
    voltage). Returns (n_bus, 2): [p_ed (pu), position in the box], zero off
    ``eligible``.
    """
    dtype = torch.float64
    c1 = c1_bus[eligible].to(dtype)
    c2 = c2_bus[eligible].to(dtype).clamp_min(1e-9)
    lo = p_min[eligible].to(dtype)
    hi = p_max[eligible].to(dtype)
    n_bus = c1_bus.shape[0]
    out = torch.zeros(n_bus, 2, dtype=dtype, device=c1_bus.device)
    if int(eligible.sum()) == 0 or p_load_total <= 0:
        return out

    def dispatch(lam):
        return ((lam - c1) / (2.0 * c2)).clamp(lo, hi)

    lam_lo = float((c1 + 2.0 * c2 * lo).min())
    lam_hi = float((c1 + 2.0 * c2 * hi).max())
    # A tight grid may exceed total Pmax; the prior then saturates at Pmax.
    target = min(max(p_load_total, float(lo.sum())), float(hi.sum()))
    for _ in range(n_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        if float(dispatch(lam).sum()) < target:
            lam_lo = lam
        else:
            lam_hi = lam
    p_ed = dispatch(0.5 * (lam_lo + lam_hi))
    width = (hi - lo).clamp_min(1e-9)
    out[eligible, 0] = p_ed
    out[eligible, 1] = (p_ed - lo) / width
    return out
