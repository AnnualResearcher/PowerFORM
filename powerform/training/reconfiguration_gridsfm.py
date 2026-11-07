"""The reconfiguration study on the GridSFM backbone (App. D, Figure 3).

Everything but the model comes from `powerform/training/reconfiguration.py` unchanged:
pools and outages, the greedy reference, candidates, ranking hinge, optimiser groups,
training loop, snapshots and scoring, with the same defaults. The model is GridSFM with
a line-status gate and track added (`powerform/model/baselines/gridsfm_track.py`).

    python -m powerform.training.reconfiguration_gridsfm --arm pretrained --seed 0
    python -m powerform.training.reconfiguration_gridsfm --arm scratch --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from powerform.device import resolve_device
from powerform.eval.scorer import control_scores, has_reference
from powerform.experiments.reference import (GRID, K_OPEN, N_COST_FEATS, N_SCENARIOS,
                                             OUT_DIR, task_pools)
from powerform.model.baselines.gridsfm_track import FETCH, GENERALIST, attach_graphs, build
from powerform.paths import shown
from powerform.training import reconfiguration as rc

MAPS = "gridsfm"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The task and training defaults are `reconfiguration.parse_args`'s."""
    ap = argparse.ArgumentParser(description="Train the line-opening policy on GridSFM.")
    ap.add_argument("--arm", required=True, choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default=str(GENERALIST),
                    help="generalist checkpoint of the pretrained arm; default the one "
                         "the paper used for every seed")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--n-train", type=int, default=N_SCENARIOS["train"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--T", type=int, default=8, help="candidates per scenario")
    ap.add_argument("--gumbel-tau", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.075)
    ap.add_argument("--k-open", type=int, default=K_OPEN)
    ap.add_argument("--z-track", type=int, default=32,
                    help="width of the per-line latent of the line-status track")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=2000,
                    help="steps between resume snapshots; 0 = none")
    ap.add_argument("--resume", action="store_true",
                    help="continue from this run's snapshot in --out-dir")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing a finished record")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    args.maps = MAPS
    return args


def load_task(args, device) -> tuple:
    """The study's pools, each instance with its GridSFM graph, outage removed."""
    train_dev, train_cpu, train_out = task_pools("train", args.n_train, N_COST_FEATS, device)
    val_dev, val_cpu, val_out = task_pools("val", N_SCENARIOS["val"], N_COST_FEATS, device)
    attach_graphs(train_dev, GRID, "train", train_out)
    attach_graphs(val_dev, GRID, "val", val_out)
    return train_dev, train_cpu, val_dev, val_cpu, val_out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{MAPS}_{args.arm}_seed{args.seed}"
    dest = out_dir / f"{stem}.json"
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{shown(dest)} exists; pass --overwrite to replace it")
    if args.arm == "pretrained" and not Path(args.init).exists():
        raise SystemExit(f"--init {args.init} not found; fetch it with: {FETCH}")

    print("loading the pools and placing their outages ...", flush=True)
    train_dev, train_cpu, val_dev, val_cpu, outages = load_task(args, device)
    ref = rc.reference_for(val_cpu, args.k_open, outages, out_dir)
    if ref["outage_lines"] != outages:
        raise SystemExit("the reference table was built on other outages")
    controls = control_scores(ref)
    print(f"controls: all-closed gap {controls['all_closed_gap_pct']:+.3f}%, "
          f"random-{args.k_open} gap {controls['random_k_gap_pct']:+.3f}%", flush=True)

    model = build(args.arm, args.z_track, Path(args.init)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{stem}: {params:,} parameters on {device}", flush=True)
    trainable, opt, sched = rc.make_optimizer(model, args)

    rng = np.random.default_rng(args.seed)
    snapshot = out_dir / f"{stem}.resume.pt"
    state = {"step": 0, "buffer": {}, "rows": [], "elapsed_s": 0.0}
    resumed_at = None
    if args.resume and snapshot.exists():
        blob = rc.resume(snapshot, model, opt, sched, rng,
                         {k: getattr(args, k) for k in rc.RESUME_MUST_MATCH})
        state = {k: blob[k] for k in state}
        resumed_at = state["step"]
        print(f"resumed from step {resumed_at}", flush=True)
    rc.train(model, opt, sched, trainable,
             rc.Task(train_dev, train_cpu, val_dev, val_cpu, ref), args, rng, state, snapshot)

    dest.write_text(json.dumps({
        "maps": MAPS, "arm": args.arm, "seed": args.seed, "grid": GRID,
        "steps": args.steps, "k_open": args.k_open, "n_train": len(train_dev.insts),
        "n_val": len(val_dev.insts),
        "n_scored": sum(has_reference(r) for r in ref["rows"]), "batch": args.batch,
        "T": args.T, "gumbel_tau": args.gumbel_tau, "margin": args.margin,
        "lr": args.lr, "trunk_lr_mult": args.trunk_lr_mult, "z_track": args.z_track,
        "init": None if args.arm == "scratch" else Path(args.init).name,
        "params": params, "resumed_from_step": resumed_at, "elapsed_s": state["elapsed_s"],
        "controls": controls, "rows": state["rows"]}, indent=1))
    ckpt = out_dir / f"reconfig_{stem}.pt"
    torch.save({"model": model.state_dict(), "config": {"arch": MAPS, "z_d": args.z_track}}, ckpt)
    snapshot.unlink(missing_ok=True)
    print(f"-> {shown(dest)}, {shown(ckpt)}", flush=True)


if __name__ == "__main__":
    main()
