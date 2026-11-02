"""Exact allocation of bus-level real power onto generators.

The model and the power flow work per bus, while the corpus labels and the cost
function are per generator.
"""
from __future__ import annotations

import torch

from powerform.data.opfdata_case import allocate_bus_p_to_gens


def max_gens_per_bus(labels, n_bus: int) -> int:
    return int(torch.bincount(labels.gen_bus, minlength=n_bus).max())


def allocate_bus_p_exact(p_bus: torch.Tensor, labels, n_bus: int | None = None):
    """Bus total -> per-generator dispatch, exact when one unit owns the bus.

    With a single unit at a bus the equal-incremental-cost solution is the bus
    total clipped to that unit's box, so the bisection is replaced by its closed
    form.
    """
    n = p_bus.numel() if n_bus is None else n_bus
    if max_gens_per_bus(labels, n) <= 1:
        return p_bus[labels.gen_bus].clamp(min=labels.pmin_gen,
                                           max=labels.pmax_gen)
    return allocate_bus_p_to_gens(p_bus, labels)
