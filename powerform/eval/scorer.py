"""Scoring a decoded line opening: one function for every arm and seed.

`score` reads the log-gates, opens the three lowest lines and asks the power flow
what that costs against the greedy reference. It never touches a loss, so it
cannot favour the objective an arm was trained on.
"""
from __future__ import annotations

import numpy as np
import torch

from powerform.data.binding import reconfiguration_binding
from powerform.physics.feasibility import open_lines, solve_loss


def topk_action(logits: torch.Tensor, k: int) -> tuple:
    """The k lines to open: the k lowest log-gates."""
    return tuple(sorted(torch.topk(-logits, k).indices.tolist()))


def vm_box_of(pool_cpu, i: int) -> tuple:
    """The voltage box of scenario ``i``, as the corpus ships it."""
    spec = pool_cpu.specs[i]
    return (spec.vm_min, spec.vm_max)


def eval_action(inst_cpu, lab_cpu, action, vm_box) -> float | None:
    """Realised AC loss (pu) after opening ``action``, or None if it is rejected.

    A candidate is rejected when the power flow does not converge or a bus leaves
    its voltage box. It has no physical realisation and hence no loss to compare.
    """
    if not action:
        return solve_loss(inst_cpu, lab_cpu, vm_box)
    return solve_loss(open_lines(inst_cpu, tuple(sorted(action))), lab_cpu, vm_box)


def has_reference(row: dict) -> bool:
    """False for a scenario in which the greedy search found no feasible opening."""
    return row["ref_pu"] is not None


@torch.no_grad()
def score(model, pool_dev, pool_cpu, ref: dict, k: int) -> dict:
    """Median and mean loss gap to the reference, and the infeasible fraction.

    Scenarios without a reference are skipped. A rejected decode is charged the
    all-closed loss, the cost of having to do nothing, and is counted separately
    so the average cannot hide it.
    """
    model.eval()
    gaps, n_rejected = [], 0
    for i, (inst_d, spec_d, lab_d) in enumerate(
            zip(pool_dev.insts, pool_dev.specs, pool_dev.labels)):
        row = ref["rows"][i]
        if not has_reference(row):
            continue
        inst2, spec2 = reconfiguration_binding(inst_d, lab_d, spec_d)
        action = topk_action(model(inst2, spec2), k)
        loss = eval_action(pool_cpu.insts[i], pool_cpu.labels[i], action,
                           vm_box_of(pool_cpu, i))
        if loss is None:
            n_rejected += 1
            loss = row["base_pu"]
        gaps.append((loss - row["ref_pu"]) / abs(row["ref_pu"]) * 100.0)
    model.train()
    return {"gap_pct": float(np.mean(gaps)),
            "gap_pct_median": float(np.median(gaps)),
            "infeasible_fraction": n_rejected / max(len(gaps), 1)}


def control_scores(ref: dict) -> dict:
    """All-closed and random-k gaps, in the same units as the model's gap."""
    closed, rand = [], []
    for row in ref["rows"]:
        if not has_reference(row):
            continue
        closed.append((row["base_pu"] - row["ref_pu"]) / abs(row["ref_pu"]) * 100.0)
        if row["random_mean_pu"] is not None:
            rand.append((row["random_mean_pu"] - row["ref_pu"])
                        / abs(row["ref_pu"]) * 100.0)
    return {"all_closed_gap_pct": float(np.mean(closed)),
            "random_k_gap_pct": float(np.mean(rand)) if rand else None}
