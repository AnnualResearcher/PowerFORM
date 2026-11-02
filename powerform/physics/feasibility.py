"""Topology edit and loss evaluation for line opening.

The expert loop, the scorer and the greedy reference all score candidates
through ``solve_loss``, so they share one feasibility rule: a candidate is
feasible if the AC power flow converges and every bus voltage lies in its box.
"""
from __future__ import annotations

import copy

import torch

from powerform.physics.newton_solver import transmission_pf_newton_pv

# Passed in place of a box to score a converged solution without asking whether
# its voltages are inside their limits. A string rather than None, so that a
# caller who forgot the box cannot land here by accident.
VM_UNCHECKED = "unchecked"

# Slack on the voltage comparison. OPF labels bind at the voltage limit, and
# re-solving a label on its own topology reproduces it with a solver error near
# 1e-8, which must not read as a violation. 1e-6 matches the residual the power
# flow is accepted at, and stays well below a real post-contingency excursion.
_VM_TOL = 1e-6


def open_lines(inst, drop: tuple[int, ...]):
    """A copy of the instance with the ``drop`` branches removed.

    An opened line is absent from the network. Faking absence by scaling a
    parameter would leave the branch in the admittance matrix, where its
    charging susceptance still injects reactive power.
    """
    dev = inst.tgrid.br_r.device
    keep = torch.ones(inst.n_edges, dtype=torch.bool, device=dev)
    for d in drop:
        keep[d] = False
    i2 = copy.copy(inst)
    tg = copy.copy(inst.tgrid)
    for f in ("br_r", "br_x", "br_b", "tap", "shift_rad", "f_bus", "t_bus"):
        setattr(tg, f, getattr(tg, f)[keep].clone())
    i2.tgrid = tg
    i2.edge_uv = torch.stack([tg.f_bus, tg.t_bus], dim=1)
    for f in ("edge_r", "edge_x", "s_max"):
        setattr(i2, f, getattr(i2, f)[keep].clone())
    i2.n_edges = int(keep.sum())
    return i2


def solve_loss(inst, lab, vm_box) -> float | None:
    """Total real loss (pu) at the label dispatch on this topology.

    The loss is total injection minus total load, read off the converged
    solution. Returns None when the candidate is rejected.

    ``vm_box`` is (vm_min, vm_max) per bus, or VM_UNCHECKED to score convergence
    alone. It has no default and None raises: a forgotten box and a deliberate
    one must not look alike.
    """
    if vm_box is None:
        raise ValueError(
            "solve_loss requires vm_box: pass (vm_min, vm_max) to enforce the "
            "voltage box, or VM_UNCHECKED to score convergence alone")
    # The slack bus is excluded from the fixed injection: the power flow derives
    # its generation, and leaving the label's slack output in p_bus would add it
    # a second time.
    ns = ~inst.tgrid.is_slack
    p_bus = torch.where(inst.m_free & ns, lab.pg_bus,
                        torch.zeros_like(lab.pg_bus))
    vm_set = torch.where(inst.m_free, lab.vm, torch.ones_like(lab.vm))
    pv = inst.m_free & ns
    try:
        pf = transmission_pf_newton_pv(inst.tgrid, p_bus, vm_set, pv, n_iter=40)
    except Exception:
        return None
    if float(pf["residual"]) > 1e-6:
        return None
    # The box tests the solved voltages, not the setpoints: a switching action
    # does its damage at the PQ buses, where no setpoint exists.
    if not isinstance(vm_box, str):
        vm_lo, vm_hi = vm_box
        vm = torch.abs(torch.as_tensor(pf["V"]))
        vm_lo = torch.as_tensor(vm_lo, dtype=vm.dtype).reshape(-1)
        vm_hi = torch.as_tensor(vm_hi, dtype=vm.dtype).reshape(-1)
        if bool(((vm < vm_lo - _VM_TOL) | (vm > vm_hi + _VM_TOL)).any()):
            return None
    slack = torch.as_tensor(pf["p_slack"], dtype=torch.float64)
    gen_total = float(p_bus.sum()) + float(slack.sum())
    return gen_total - float(inst.p_load.sum())
