"""Binding a corpus instance to the reconfiguration task.

The model is told which variables a task leaves free through a TaskSpec. Here
the dispatch is fixed at the label and the gate z is the decision. The fixed
values travel on the channels that carried the same kind of quantity during OPF
pretraining, so no pretrained encoder meets an input it was not trained on.
"""
from __future__ import annotations

import copy

import torch

from powerform.model.spec import TaskSpec


def reconfiguration_binding(inst, lab, spec0):
    """Re-bind one scenario from OPF roles to power-flow roles.

    Returns a shallow copy of the instance plus its spec, so the pool is never
    mutated.

    Loads stay as OPF training saw them, non-negative: net demand would be a
    different input distribution with a mirrored tail. The generator setpoint
    goes on p_max, which carried capacity, the same kind of quantity in the same
    units; p_min stays at the constant 0 it always was. The setpoint is clamped
    at zero because the raw label carries signed-zero values near -1e-8.
    """
    slack = inst.tgrid.is_slack
    gen = (lab.pg_bus != 0) | (lab.qg_bus != 0)
    gen = gen & ~slack

    give = (~slack).to(lab.pg_bus.dtype)
    inst2 = copy.copy(inst)
    pg_given = lab.pg_bus * give

    # |V| box: the bus's own box at PQ buses, pinned to the setpoint at
    # generator buses.
    vmin = torch.where(gen, lab.vm, spec0.vm_min)
    vmax = torch.where(gen, lab.vm, spec0.vm_max)
    pg_given = pg_given.clamp_min(0.0)
    pmin = torch.zeros_like(pg_given)
    pmax = pg_given
    spec = TaskSpec(
        m_node=~slack,
        m_edge=torch.zeros(inst.n_edges, dtype=torch.bool,
                           device=inst.p_load.device),
        p_min=pmin, p_max=pmax,
        vm_min=vmin, vm_max=vmax,
    )
    return inst2, spec
