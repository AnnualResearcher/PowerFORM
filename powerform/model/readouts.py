"""Decision readout: hidden state -> (p_g, vm_set), box-constrained.

Each decision is read from the bus variable it physically belongs to, active
power from the injection stream and the voltage setpoint from the voltage
stream, and squashed into the task's own box.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from powerform.model.spec import TaskSpec


class DecisionReadout(nn.Module):
    """``merged=True`` is the merged-readout variant: both decisions read the
    whole bus state [S, V], at half the hidden width, so the parameter count
    stays the same."""

    def __init__(self, d: int, merged: bool = False):
        super().__init__()
        self.merged = bool(merged)
        d_in = 2 * d if self.merged else d
        h = d // 2 if self.merged else d
        self.net_p = nn.Sequential(
            nn.Linear(d_in + 2, h), nn.GELU(), nn.Linear(h, 1))
        self.net_v = nn.Sequential(
            nn.Linear(d_in + 2, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, S_lat, V_lat, spec: TaskSpec):
        src_p = torch.cat([S_lat, V_lat], dim=-1) if self.merged else S_lat
        src_v = torch.cat([S_lat, V_lat], dim=-1) if self.merged else V_lat
        xp = torch.cat([src_p, spec.p_min.unsqueeze(-1),
                        spec.p_max.unsqueeze(-1)], dim=-1)
        xv = torch.cat([src_v, spec.vm_min.unsqueeze(-1),
                        spec.vm_max.unsqueeze(-1)], dim=-1)
        p_g = spec.p_min + (spec.p_max - spec.p_min) * torch.sigmoid(
            self.net_p(xp)[:, 0])
        vm = spec.vm_min + (spec.vm_max - spec.vm_min) * torch.sigmoid(
            self.net_v(xv)[:, 0])
        zero = torch.zeros_like(p_g)
        return (torch.where(spec.m_node, p_g, zero),
                torch.where(spec.m_node, vm, zero))
