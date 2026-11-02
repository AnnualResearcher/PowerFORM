"""Task binding and the per-instance input standardisation.

A TaskSpec holds no learnable content. It says which variables a task leaves
free and the boxes they live in, so the same trunk serves another task by being
handed another spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-8


@dataclass
class TaskSpec:
    """Role binding of one instance.

    m_node : (N,) bool   True where the decisions (p_g, vm_set) are free
    m_edge : (E,) bool   True where the line gate is a free decision; a
                         constant input channel of the line embedding
    p_min, p_max : (N,)  active-power box, read where m_node
    vm_min, vm_max : (N,) voltage-setpoint box, read where m_node
    """
    m_node: torch.Tensor
    m_edge: torch.Tensor
    p_min: torch.Tensor
    p_max: torch.Tensor
    vm_min: torch.Tensor
    vm_max: torch.Tensor


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit standard deviation over one instance's buses."""
    return (x - x.mean()) / (x.std() + _EPS)
