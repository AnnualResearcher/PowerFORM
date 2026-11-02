"""PV-bus Newton power flow, the physics every result is measured against.

The model proposes setpoints; this solve decides what the network does with
them. The same solver labels expert candidates during training, scores decoded
actions at evaluation and builds the greedy reference, so the three cannot
disagree about what a solution costs. The branch model is MATPOWER's
pi-equivalent with tap ratio and phase shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransmissionGrid:
    """Network and loads of one instance. Branch arrays follow the MATPOWER
    branch table and hold in-service branches only."""
    f_bus: torch.Tensor        # (E,) int64
    t_bus: torch.Tensor        # (E,) int64
    br_r: torch.Tensor         # (E,) pu
    br_x: torch.Tensor         # (E,) pu
    br_b: torch.Tensor         # (E,) total line charging susceptance (pu)
    tap: torch.Tensor          # (E,) tap magnitude (1.0 for lines)
    shift_rad: torch.Tensor    # (E,) phase shift (rad)
    gs: torch.Tensor           # (N,) bus shunt conductance (pu)
    bs: torch.Tensor           # (N,) bus shunt susceptance (pu)
    p_load: torch.Tensor       # (N,) pu
    q_load: torch.Tensor       # (N,) pu
    is_slack: torch.Tensor     # (N,) bool

    @property
    def n_nodes(self) -> int:
        return int(self.p_load.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.f_bus.shape[0])


def build_ybus(g: TransmissionGrid) -> torch.Tensor:
    """Full pi-model Ybus (N, N), complex128."""
    n = g.n_nodes
    cdtype = torch.complex128

    y = 1.0 / (g.br_r.to(cdtype) + 1j * g.br_x.to(cdtype))
    bc2 = 1j * g.br_b.to(cdtype) / 2.0
    t = g.tap.to(cdtype) * torch.exp(1j * g.shift_rad.to(cdtype))
    tau2 = (g.tap.to(cdtype)) ** 2

    y_ff = (y + bc2) / tau2
    y_ft = -y / t.conj()
    y_tf = -y / t
    y_tt = y + bc2

    Y = torch.zeros(n, n, dtype=cdtype, device=g.br_r.device)
    f, tt = g.f_bus, g.t_bus
    Y.index_put_((f, f), y_ff, accumulate=True)
    Y.index_put_((f, tt), y_ft, accumulate=True)
    Y.index_put_((tt, f), y_tf, accumulate=True)
    Y.index_put_((tt, tt), y_tt, accumulate=True)

    sh = g.gs.to(cdtype) + 1j * g.bs.to(cdtype)
    idx = torch.arange(n, device=g.br_r.device)
    Y.index_put_((idx, idx), sh, accumulate=True)
    return Y


def transmission_pf_newton_pv(
    g: TransmissionGrid,
    p_g: torch.Tensor,          # (N,) bus gen P (pu); slack P derived
    vm_set: torch.Tensor,       # (N,) voltage setpoints (used at PV+slack)
    pv_mask: torch.Tensor,      # (N,) bool: PV buses (Vm fixed, Q free)
    n_iter: int = 40,
    tol: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Polar Newton AC power flow with PV buses. The decision variables are
    (P_g, V_setpoint); reactive dispatch at PV and slack buses is derived from
    the solution.

    unknowns  : Va at non-slack, Vm at PQ buses
    mismatch  : P at non-slack, Q at PQ buses
    Vm pinned : slack + PV buses at vm_set; slack angle = 0.

    Returns V, the largest residual on the solved equations, the derived q_g
    and the per-bus p_slack.
    """
    n = g.n_nodes
    cdtype = torch.complex128
    Y = build_ybus(g)

    slack = g.is_slack
    pv = pv_mask & ~slack
    pq = ~slack & ~pv
    ns_idx = torch.where(~slack)[0]          # P-mismatch rows / Va cols
    pq_idx = torch.where(pq)[0]              # Q-mismatch rows / Vm cols
    n_ns = len(ns_idx)

    # q_load enters the specification only through the Q rows of the PQ buses
    S_spec = -(((g.p_load - p_g) + 1j * (g.q_load - 0.0)).to(cdtype))

    def mismatch(Vm_, Va_):
        V_ = (Vm_ * torch.exp(1j * Va_)).to(cdtype)
        S_ = V_ * (Y @ V_).conj()
        m_ = S_ - S_spec
        return torch.cat([m_.real[ns_idx], m_.imag[pq_idx]])

    # flat start: Va = 0, Vm = setpoints at PV and slack, 1.0 elsewhere
    Vm = torch.where(pv | slack, vm_set,
                     torch.ones(n, dtype=torch.float64, device=Y.device))
    Va = torch.zeros(n, dtype=torch.float64, device=Y.device)

    for _ in range(n_iter):
        V = (Vm * torch.exp(1j * Va)).to(cdtype)
        I = Y @ V
        F = mismatch(Vm, Va)
        if float(F.abs().max()) < tol:
            break

        diagV = torch.diag(V)
        diagI = torch.diag(I)
        diagVnorm = torch.diag((V / V.abs().to(cdtype)))
        dS_dVa = 1j * diagV @ (diagI - Y @ diagV).conj()
        dS_dVm = diagVnorm @ diagI.conj() + diagV @ (Y @ diagVnorm).conj()

        J11 = dS_dVa.real[ns_idx][:, ns_idx]          # dP/dVa
        J12 = dS_dVm.real[ns_idx][:, pq_idx]          # dP/dVm(PQ)
        J21 = dS_dVa.imag[pq_idx][:, ns_idx]          # dQ(PQ)/dVa
        J22 = dS_dVm.imag[pq_idx][:, pq_idx]          # dQ(PQ)/dVm(PQ)
        J = torch.cat([
            torch.cat([J11, J12], dim=1),
            torch.cat([J21, J22], dim=1),
        ], dim=0)

        dx = torch.linalg.solve(J, -F)

        f0 = float(F.norm())
        t = 1.0
        accepted = False
        for _ls in range(12):
            Va_t = Va.index_add(0, ns_idx, t * dx[:n_ns])
            Vm_t = Vm.index_add(0, pq_idx, t * dx[n_ns:]).clamp_min(0.2)
            if float(mismatch(Vm_t, Va_t).norm()) < f0:
                accepted = True
                break
            t *= 0.5
        if accepted:
            Va, Vm = Va_t, Vm_t
        else:
            break

    V = (Vm * torch.exp(1j * Va)).to(cdtype)
    S_calc = V * (Y @ V).conj()
    resid = torch.cat([
        (S_calc.real - S_spec.real)[ns_idx],
        (S_calc.imag - S_spec.imag)[pq_idx]]).abs().max()

    q_g_derived = torch.where(pv | slack,
                              S_calc.imag + g.q_load,
                              torch.zeros_like(g.q_load))
    p_slack = torch.where(slack, S_calc.real + g.p_load,
                          torch.zeros_like(g.p_load))
    return {"V": V, "residual": resid, "q_g": q_g_derived, "p_slack": p_slack}
