"""Why decoded openings are rejected: divergence, or the voltage box.

The infeasible fraction of a run counts both causes. This decodes a final
checkpoint on the scored validation scenarios (those with a feasible reference,
so the fraction here equals the run's) and reports which branch rejected each.

    python -m powerform.eval.infeasible_split --arm pretrained --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from powerform.data.binding import reconfiguration_binding
from powerform.device import resolve_device
from powerform.eval.checkpoint import load_model
from powerform.eval.scorer import has_reference, topk_action, vm_box_of
from powerform.experiments.reference import K_OPEN, N_SCENARIOS, OUT_DIR, task_pools
from powerform.model.line_track import ReconfigurationNet
from powerform.paths import CHECKPOINTS, RESULTS, shown
from powerform.physics.feasibility import _VM_TOL, open_lines
from powerform.physics.newton_solver import transmission_pf_newton_pv


def reason_for(inst, lab, vm_box) -> str:
    """'ok', 'diverged' or 'out_of_box': the computation of `solve_loss`, reporting
    the branch that rejects instead of collapsing them."""
    ns = ~inst.tgrid.is_slack
    p_bus = torch.where(inst.m_free & ns, lab.pg_bus, torch.zeros_like(lab.pg_bus))
    vm_set = torch.where(inst.m_free, lab.vm, torch.ones_like(lab.vm))
    pv = inst.m_free & ns
    try:
        pf = transmission_pf_newton_pv(inst.tgrid, p_bus, vm_set, pv, n_iter=40)
    except Exception:
        return "diverged"
    if float(pf["residual"]) > 1e-6:
        return "diverged"
    vm = torch.abs(torch.as_tensor(pf["V"]))
    lo = torch.as_tensor(vm_box[0], dtype=vm.dtype).reshape(-1)
    hi = torch.as_tensor(vm_box[1], dtype=vm.dtype).reshape(-1)
    if bool(((vm < lo - _VM_TOL) | (vm > hi + _VM_TOL)).any()):
        return "out_of_box"
    return "ok"


def split_of(net, pool_dev, pool_cpu, ref: dict) -> dict:
    """Counts of each reason over the scored scenarios."""
    scored = [i for i, r in enumerate(ref["rows"]) if has_reference(r)]
    counts = {"ok": 0, "diverged": 0, "out_of_box": 0}
    with torch.no_grad():
        for i in scored:
            inst2, spec2 = reconfiguration_binding(
                pool_dev.insts[i], pool_dev.labels[i], pool_dev.specs[i])
            action = topk_action(net(inst2, spec2), ref["k"])
            counts[reason_for(open_lines(pool_cpu.insts[i], action),
                              pool_cpu.labels[i], vm_box_of(pool_cpu, i))] += 1
    rejected = counts["diverged"] + counts["out_of_box"]
    return {**counts, "n_scored": len(scored), "n_rejected": rejected,
            "infeasible_fraction": rejected / len(scored),
            "diverged_share_of_rejections": (counts["diverged"] / rejected
                                             if rejected else None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--maps", default="physics", choices=["physics", "generic"])
    ap.add_argument("--arm", required=True, choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", default=None,
                    help="default checkpoints/reconfig_<maps>_<arm>_seed<seed>.pt")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results tree holding reconfiguration/reference_val.json")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else (
        CHECKPOINTS / f"reconfig_{args.maps}_{args.arm}_seed{args.seed}.pt")
    ref_path = args.results / "reconfiguration" / "reference_val.json"
    if not ref_path.exists():
        raise SystemExit(f"no reference table at {ref_path}; build one with "
                         "python -m powerform.experiments.reference --split val")
    net = load_model(ckpt, device)
    if not isinstance(net, ReconfigurationNet):
        raise SystemExit(f"{ckpt} has no line-status track; this reads the "
                         "reconfiguration checkpoints")
    pool_dev, pool_cpu, _ = task_pools("val", N_SCENARIOS["val"],
                                       net.trunk.n_cost_feats, device)
    ref = json.loads(ref_path.read_text())
    if ref["k"] != K_OPEN or ref["n"] != len(pool_cpu.insts):
        raise SystemExit(f"the reference table has k={ref['k']}, n={ref['n']}; "
                         f"this task has k={K_OPEN}, n={len(pool_cpu.insts)}")
    row = split_of(net, pool_dev, pool_cpu, ref)
    print(f"ok {row['ok']}, diverged {row['diverged']}, out of box "
          f"{row['out_of_box']} of {row['n_scored']}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"infeasible_split_{args.maps}_{args.arm}_seed{args.seed}.json"
    out.write_text(json.dumps(
        {"maps": args.maps, "arm": args.arm, "seed": args.seed, "grid": ref["grid"],
         "n_val": ref["n"], "vm_tol": _VM_TOL, **row}, indent=1))
    print(f"-> {shown(out)}", flush=True)


if __name__ == "__main__":
    main()
