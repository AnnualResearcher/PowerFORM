"""Score an OPF checkpoint on a grid's OPFData test split.

Every Table 2 cell is one (checkpoint, grid, topology) evaluation. A cell carries
the cost MAPE after the terminal exact PV Newton solve: the decision the model
makes (bus Pg, Vset) carried through physics, with the reference bus's active
power supplied by the solve. Scenarios whose solve does not reach a 1e-6 pu
residual are counted and left out of the mean. A cell where every scenario fails
is reported as such, never as a crash. The state came out of the solve and so
satisfies KCL, which makes each violation axis (`eval/violations.py`) a real one.
The state channels are for reference: the model decides setpoints, it does not
regress a state.

`--baseline ed` scores training-free economic dispatch through the same protocol.

    python -m powerform.eval.opf_eval --ckpt checkpoints/opf_generalist_physics_seed0.pt \\
        --eval case57,case118,case500,case30
    python -m powerform.eval.opf_eval --baseline ed --eval case57 --variant n1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from powerform.data.allocation import allocate_bus_p_exact
from powerform.data.opfdata import ALIAS_OF, CASE_ALIAS, SPLIT_BOUNDS, load_pool, resolve_case
from powerform.data.opfdata_case import (UNRATED_SENTINEL, allocate_bus_q_to_gens,
                                         gen_cost)
from powerform.device import resolve_device
from powerform.eval.checkpoint import load_model
from powerform.eval.violations import TOL, audit_state
from powerform.model.powerform import PowerForm
from powerform.paths import OUTPUTS, shown
from powerform.physics.newton_solver import (build_ybus,
                                             transmission_pf_newton_pv)
from powerform.physics.pf_pv_sparse import build_ybus_sparse, pf_pv_sparse

OUT_DIR = OUTPUTS / "table2"
ED_N_COST_FEATS = 12            # the prior lives in cost-feature columns 10-11
# Above this size a dense Jacobian and Ybus cost too much memory.
SPARSE_ABOVE_N = 1000


def scenario_statics(inst, spec, lab) -> dict:
    """`violations.audit_state`'s static block, built from OPFData.

    `spec.p_min/p_max` are zeroed on the reference bus because the model does not
    decide slack power, but the derived-P axis asks about the physical box. So the
    P and Q boxes are summed from the per-generator limits on the labels.
    """
    n = inst.n_nodes
    gen_bus = lab.gen_bus

    def to_bus(vals: torch.Tensor) -> np.ndarray:
        acc = torch.zeros(n, dtype=torch.float64)
        acc.index_add_(0, gen_bus.cpu(), vals.cpu())
        return acc.numpy()

    return {
        "pd": inst.p_load.cpu().numpy(),
        "qd": inst.q_load.cpu().numpy(),
        "vmin": spec.vm_min.cpu().numpy(),
        "vmax": spec.vm_max.cpu().numpy(),
        "free": inst.m_free.cpu().numpy(),
        "qmin": to_bus(lab.qmin_gen), "qmax": to_bus(lab.qmax_gen),
        "pmin": to_bus(lab.pmin_gen), "pmax": to_bus(lab.pmax_gen),
    }


def ed_decision(inst, lab) -> tuple:
    """Training-free economic dispatch, as a decision. The closed-form prior ignores
    the network, and flat 1.0 setpoints keep the labels out of the baseline."""
    p_bus = torch.where(inst.m_node_ns, inst.cost_feats[:, 10],
                        torch.zeros_like(lab.pg_bus))
    return p_bus, torch.ones_like(lab.vm)


def model_decision(model, inst, spec) -> tuple:
    """The readouts already sigmoid into each bus's own voltage box."""
    out = model(inst, spec)
    vm_set = torch.where(inst.m_free, out["vm_set"], torch.ones_like(out["vm_set"]))
    p_bus = torch.where(inst.m_free, out["p_g"], torch.zeros_like(out["p_g"]))
    return p_bus, vm_set


def score_scenario(inst, spec, lab, p_bus, vm_set, use_sparse: bool) -> tuple | None:
    """Decision -> solve -> per-scenario metrics; None if the solve failed.
    Returns (violation record, state and cost channels)."""
    pv = inst.m_free & ~inst.tgrid.is_slack
    if use_sparse:
        pf = pf_pv_sparse(inst.tgrid, p_bus, vm_set, pv)
    else:
        pf = transmission_pf_newton_pv(inst.tgrid, p_bus, vm_set, pv, n_iter=40)
    if float(pf["residual"]) > 1e-6:
        return None

    # The dense solver returns a tensor, the sparse one a numpy array.
    V = pf["V"]
    V = V.detach().cpu().numpy() if torch.is_tensor(V) else np.asarray(V)
    vm, va = np.abs(V), np.angle(V)
    grid = inst.tgrid
    Y = (build_ybus_sparse(grid) if use_sparse
         else build_ybus(grid).detach().cpu().numpy())
    rate = inst.s_max.cpu().numpy()

    dev = p_bus.device
    p_final = p_bus.detach() + torch.as_tensor(pf["p_slack"], dtype=torch.float64,
                                               device=dev)
    pg_gen = allocate_bus_p_exact(p_final, lab, inst.n_nodes)
    qg_gen = allocate_bus_q_to_gens(
        torch.as_tensor(pf["q_g"], dtype=torch.float64, device=dev), lab)
    cost_pred = float(gen_cost(pg_gen, lab.c2, lab.c1, lab.c0))
    cost_gt = lab.cost

    rated = rate < UNRATED_SENTINEL
    rec = audit_state(vm, va, scenario_statics(inst, spec, lab), grid, Y,
                      np.where(rated, rate, np.inf))
    sidx = int(torch.where(grid.is_slack)[0][0])
    vm_lab, va_lab = lab.vm.cpu().numpy(), lab.va.cpu().numpy()
    va_z, va_lab_z = va - va[sidx], va_lab - va_lab[sidx]
    return rec, {
        "cost_mape": abs(cost_pred - cost_gt) / max(abs(cost_gt), 1.0),
        "cost_rel_signed": (cost_pred - cost_gt) / max(abs(cost_gt), 1.0),
        "vm_mae": float(np.mean(np.abs(vm - vm_lab))),
        "v_mape_pct": float(np.mean(np.abs(vm - vm_lab) / np.abs(vm_lab)) * 100.0),
        "va_mae_deg": float(np.rad2deg(np.mean(np.abs(va_z - va_lab_z)))),
        "pg_mae": float(torch.mean((pg_gen - lab.pg_gen).abs())),
        "qg_mae": float(torch.mean((qg_gen - lab.qg_gen).abs())),
    }


def violation_axes(recs: list) -> dict:
    """The four violation axes as per-scenario means and run maxima."""
    def mean(key):
        return float(np.mean([r[key] for r in recs]))

    def axis(n_key, max_key, max_name):
        return {"per_scenario_mean_count": mean(n_key),
                "scenarios_with_any": int(sum(r[n_key] > 0 for r in recs)),
                max_name: float(np.max([r[max_key] for r in recs]))}

    return {
        "voltage_box": axis("v_n", "v_max", "worst_pu"),
        "derived_Q": axis("q_n", "q_max", "worst_pu"),
        "derived_P": {**axis("p_n", "p_max", "worst_pu"),
                      "per_scenario_mean_count_slack": mean("p_n_slack")},
        "thermal": axis("n_over", "load_max", "worst_loading"),
        "violated_inequalities_per_scenario": float(np.mean(
            [r["v_n"] + r["q_n"] + r["n_over"] + r["p_n_slack"] for r in recs])),
        "tolerance_pu": TOL,
    }


@torch.no_grad()
def eval_grid(model, pool, baseline: str | None = None) -> dict:
    """One cell: every scenario of `pool` through decision -> solve -> score."""
    recs, chans, n_failed = [], [], 0
    use_sparse = pool.insts[0].n_nodes > SPARSE_ABOVE_N
    for k, (inst, spec, lab) in enumerate(zip(pool.insts, pool.specs, pool.labels)):
        p_bus, vm_set = (ed_decision(inst, lab) if baseline == "ed"
                         else model_decision(model, inst, spec))
        scored = score_scenario(inst, spec, lab, p_bus, vm_set, use_sparse)
        if (k + 1) % 50 == 0:
            print(f"  {pool.alias}: {k + 1}/{len(pool)}", flush=True)
        if scored is None:
            n_failed += 1
            continue
        recs.append(scored[0])
        chans.append(scored[1])
    if not chans:
        return {"n_eval": len(pool), "n_scored": 0, "n_nonconverged": n_failed}

    def mean(key):
        return float(np.mean([c[key] for c in chans]))

    return {
        "n_eval": len(pool),
        "n_scored": len(chans),
        "n_nonconverged": n_failed,
        "cost_mape_pct": mean("cost_mape") * 100.0,
        "cost_mape_pct_median": float(np.median(
            [c["cost_mape"] for c in chans])) * 100.0,
        "cost_rel_signed_pct": mean("cost_rel_signed") * 100.0,
        "vm_mae": mean("vm_mae"), "v_mape_pct": mean("v_mape_pct"),
        "va_mae_deg": mean("va_mae_deg"),
        "pg_mae": mean("pg_mae"), "qg_mae": mean("qg_mae"),
        "violations": violation_axes(recs),
        "provenance": pool.provenance(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a checkpoint or economic dispatch on OPFData test "
                    "splits; one JSON cell per grid.")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--ckpt", help="checkpoint to score")
    who.add_argument("--baseline", choices=["ed"],
                     help="training-free economic dispatch with flat setpoints")
    ap.add_argument("--eval", required=True,
                    help="comma-separated grids, e.g. case57,case118,case2000")
    ap.add_argument("--variant", default="fulltop", choices=["fulltop", "n1"])
    ap.add_argument("--n", type=int, default=0,
                    help="evenly subsampled scenarios; 0 = the whole split")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    grids = [g.strip() for g in args.eval.split(",") if g.strip()]
    for grid in grids:
        if resolve_case(grid) not in ALIAS_OF:
            raise SystemExit(f"unknown grid {grid}; choose from {', '.join(CASE_ALIAS)}")

    device = resolve_device(args.device)
    if args.baseline:
        model, name, n_cost_feats = None, args.baseline, ED_N_COST_FEATS
    else:
        try:
            model = load_model(args.ckpt, device)
        except ValueError as e:
            raise SystemExit(f"{args.ckpt} is not a PowerForm checkpoint; this scores "
                             "OPF generalists") from e
        if not isinstance(model, PowerForm):
            raise SystemExit(f"{args.ckpt} is a reconfiguration checkpoint; this "
                             "scores OPF generalists")
        if model.n_cost_feats == 0:
            raise SystemExit(f"{args.ckpt} is a Table 1 checkpoint: it has no cost inputs, "
                             "so a score on OPFData means nothing. Re-score it with "
                             "python -m powerform.training.case500 --eval-only")
        name, n_cost_feats = model.maps, model.n_cost_feats
    stem = (args.baseline if args.baseline
            else Path(args.ckpt).stem.removeprefix("opf_generalist_"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = SPLIT_BOUNDS["test"]
    for grid in grids:
        pool = load_pool(grid, args.variant, "test", n=args.n,
                         n_cost_feats=n_cost_feats, device=device)
        outside = [i for i in pool.example_indices if not lo <= i < hi]
        if outside:
            raise SystemExit(f"eval pool for {grid} leaves the test range "
                             f"[{lo},{hi}): {outside[:5]}")
        t0 = time.time()
        cell = eval_grid(model, pool, baseline=args.baseline)
        # A subset has a file of its own, so it cannot replace a whole cell.
        subset = f"_n{args.n}" if args.n else ""
        out = out_dir / f"{stem}_{pool.alias}_{args.variant}{subset}.json"
        out.write_text(json.dumps(
            {"model": name, "checkpoint": Path(args.ckpt).name if args.ckpt else None,
             "grid": pool.alias, "variant": args.variant, **cell}, indent=2))
        if cell["n_scored"]:
            print(f"  {pool.alias}/{args.variant}: cost MAPE "
                  f"{cell['cost_mape_pct']:.4f}% | nonconverged "
                  f"{cell['n_nonconverged']}/{cell['n_eval']} "
                  f"({time.time() - t0:.0f}s) -> {shown(out)}", flush=True)
        else:
            print(f"  {pool.alias}/{args.variant}: no scored scenarios, "
                  f"{cell['n_nonconverged']}/{cell['n_eval']} failed to converge "
                  f"-> {shown(out)}", flush=True)


if __name__ == "__main__":
    main()
