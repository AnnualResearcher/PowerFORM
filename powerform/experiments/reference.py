"""The reconfiguration task: its N-1 outages and its greedy reference.

`apply_n1` places one line outage per scenario. The draw depends only on the split
and the scenario index, never on a run's seed, so every arm and seed solves the
same problems. `reference_table` searches the feasible openings of every scenario
and is what each gap is measured against.

    python -m powerform.experiments.reference --split val [--n N] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from powerform.data.opfdata import load_pool
from powerform.eval.scorer import eval_action, vm_box_of
from powerform.paths import OUTPUTS, shown
from powerform.physics.feasibility import VM_UNCHECKED, open_lines, solve_loss

GRID = "case57"
K_OPEN = 3
N_SCENARIOS = {"train": 2000, "val": 157}
N_RANDOM = 8                    # random openings per scenario, the random-k control
N_COST_FEATS = 12
OUT_DIR = OUTPUTS / "reconfiguration"


def task_pools(split: str, n: int, n_cost_feats: int, device) -> tuple:
    """A pool on the model's device and a CPU copy for the power flows, both with
    the N-1 outages applied. Returns (device pool, CPU pool, outaged lines)."""
    pool_dev = load_pool(GRID, "fulltop", split, n=n, n_cost_feats=n_cost_feats,
                         device=device, verbose=False)
    pool_cpu = load_pool(GRID, "fulltop", split, n=n, n_cost_feats=n_cost_feats,
                         verbose=False)
    for a, b in zip(pool_dev.insts, pool_cpu.insts):
        assert abs(float(a.p_load.sum()) - float(b.p_load.sum())) < 1e-9, \
            "the device and CPU copies of the pool are not in the same order"
    return pool_dev, pool_cpu, apply_n1(pool_dev, pool_cpu, split)


def apply_n1(pool_dev, pool_cpu, split: str) -> list[int]:
    """Take one line out of service in every scenario, in place, on both copies.

    The voltage box is not applied here: an outage is a contingency, not a
    decision, and one that pushes voltages out of limits is the situation
    switching exists to fix. A scenario with no connected outage among 20 draws
    stays intact and is recorded as -1.
    """
    # crc32 rather than hash(): Python randomises string hashes per process.
    rng = np.random.default_rng(zlib.crc32(split.encode()) & 0xFFFFFFFF)
    chosen: list[int] = []
    for i in range(len(pool_cpu.insts)):
        inst_c, lab_c = pool_cpu.insts[i], pool_cpu.labels[i]
        pick = None
        for _ in range(20):
            cand = int(rng.integers(inst_c.n_edges))
            if solve_loss(open_lines(inst_c, (cand,)), lab_c,
                          VM_UNCHECKED) is not None:
                pick = cand
                break
        if pick is None:
            chosen.append(-1)
            continue
        pool_cpu.insts[i] = open_lines(inst_c, (pick,))
        pool_dev.insts[i] = open_lines(pool_dev.insts[i], (pick,))
        chosen.append(pick)
    return chosen


def greedy_reference(inst_cpu, lab_cpu, k: int, vm_box) -> tuple:
    """Greedy search over openings: try every line, keep the best, repeat.

    Returns (action, loss, base loss). The search is sequential, so this is a
    strong reference rather than a global optimum, and a negative gap is possible.

    The base loss is scored without the box. It is the realised loss of doing
    nothing after the contingency, often outside the box, and a state switching
    should fix rather than a reason to drop the scenario. Without any feasible
    opening the loss is None and the scenario is not scored.
    """
    base = solve_loss(inst_cpu, lab_cpu, VM_UNCHECKED)
    if base is None:
        return (), None, None
    chosen: list[int] = []
    best = None
    for _ in range(k):
        cand_best, cand_line = None, None
        for e in range(inst_cpu.n_edges):
            if e in chosen:
                continue
            loss = eval_action(inst_cpu, lab_cpu, chosen + [e], vm_box)
            if loss is not None and (cand_best is None or loss < cand_best):
                cand_best, cand_line = loss, e
        if cand_line is None:
            break
        chosen.append(cand_line)
        best = cand_best
    return tuple(sorted(chosen)), best, base


def reference_table(pool_cpu, split: str, k: int, outages: list[int]) -> dict:
    """The reference record of a pool, with the random-k control.

    Nothing in it depends on a run's seed: the table is shared by every arm.
    """
    rng = np.random.default_rng(zlib.crc32(f"random|{split}".encode()) & 0xFFFFFFFF)
    rows = []
    for i, (inst, lab) in enumerate(zip(pool_cpu.insts, pool_cpu.labels)):
        box = vm_box_of(pool_cpu, i)
        action, best, base = greedy_reference(inst, lab, k, box)
        losses = []
        for _ in range(N_RANDOM):
            a = tuple(sorted(rng.choice(inst.n_edges, size=k,
                                        replace=False).tolist()))
            loss = eval_action(inst, lab, a, box)
            if loss is not None:
                losses.append(loss)
        rows.append({
            "i": i, "base_pu": base, "ref_action": list(action), "ref_pu": best,
            "random_mean_pu": float(np.mean(losses)) if losses else None,
            "n_random_converged": len(losses),
        })
        if len(rows) % 25 == 0:
            print(f"  reference {len(rows)}/{len(pool_cpu.insts)}", flush=True)
    return {"grid": GRID, "split": split, "k": k, "n": len(rows),
            "outage_lines": outages, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a greedy reference table.")
    ap.add_argument("--split", default="val", choices=sorted(N_SCENARIOS))
    ap.add_argument("--n", type=int, default=None,
                    help="first n scenarios of the table; default the whole pool")
    ap.add_argument("--k-open", type=int, default=K_OPEN)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    _, pool_cpu, outages = task_pools(args.split, N_SCENARIOS[args.split],
                                      N_COST_FEATS, torch.device("cpu"))
    if args.n is not None:
        pool_cpu.insts, outages = pool_cpu.insts[:args.n], outages[:args.n]
    t0 = time.time()
    doc = reference_table(pool_cpu, args.split, args.k_open, outages)
    # A subset has a file of its own, so it cannot replace the whole table.
    subset = f"_first{args.n}" if args.n is not None else ""
    out = Path(args.out_dir) / f"reference_{args.split}{subset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"-> {shown(out)}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
