"""OPF pretraining: the generalist, trained jointly across grids.

Each step draws four scenarios from the flattened pool of all training grids and
runs them as one disjoint graph. Every block is supervised toward the partial
solution t_k * label, normalised by grid size so the largest grid cannot dominate
the gradient. `--ed-prior` selects the 12-feature input the reconfiguration study
starts from; the default is the 10-feature input of the Table 2 generalists.

    python -m powerform.training.generalist --maps physics --seed 0
    python -m powerform.training.generalist --maps generic --seed 0 --ed-prior
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from powerform.data.allocation import allocate_bus_p_exact
from powerform.data.batch import merge_instances
from powerform.data.opfdata import assert_disjoint, label_cross_check, load_pool
from powerform.data.opfdata_case import gen_cost
from powerform.device import resolve_device
from powerform.eval.checkpoint import config_of
from powerform.model.powerform import PowerForm
from powerform.paths import OUTPUTS, shown
from powerform.physics.newton_solver import transmission_pf_newton_pv
from powerform.training.snapshot import save_snapshot

TRAIN_GRIDS = ("case57", "case118", "case500")
VARIANT = "fulltop"
D_MODEL, K = 216, 8
# Width of the generic block's MLPs, chosen for parameter parity with the physics
# maps.
GENERIC_HIDDEN = {"physics": None, "generic": 450}
BATCH = 4
W_BAL = 5.0
LR = 3e-4
CLIP = 1.0
SNAPSHOT_EVERY = 1000
OUT_DIR = OUTPUTS / "generalists"


def loss_from_history(hist: list, inst, spec) -> tuple:
    """The interpolant loss of one instance from its per-block readouts: block k
    is supervised toward t_k * label with t_k = (k + 1) / K, so the last block
    meets the label. Returns (loss, K)."""
    n_blocks = len(hist)
    pg_idx = torch.where(inst.m_node_ns)[0]
    vm_idx = torch.where(inst.m_free)[0]
    pg_l = inst.pg_label[pg_idx]
    vm_l = inst.vm_ref[vm_idx]

    pg_scale2 = (spec.p_max[pg_idx].mean().clamp_min(0.1)) ** 2
    np_g = max(1, len(pg_idx))
    nv = max(1, len(vm_idx))

    loss = 0.0
    for k, S_k in enumerate(hist):
        t_k = (k + 1) / n_blocks
        sup_p = ((S_k["p_g"][pg_idx] - t_k * pg_l) ** 2).sum() / (pg_scale2 * np_g)
        vm_t = 1.0 + t_k * (vm_l - 1.0)
        sup_v = ((S_k["vm_set"][vm_idx] - vm_t) ** 2).sum() * 100.0 / nv
        bal = ((S_k["p_g"][pg_idx].sum() - t_k * pg_l.sum()) ** 2
               / (pg_scale2 * np_g))
        loss = loss + sup_p + sup_v + W_BAL * bal / np_g
    return loss, n_blocks


def batch_loss(model, chosen: list) -> tuple:
    """One forward for the whole batch, each instance scored from its own slice
    of the merged readouts. Returns (loss to optimise, mean loss per block)."""
    merged, mspec = merge_instances([i for i, _ in chosen], [s for _, s in chosen])
    hist_m = model(merged, mspec)["history"]
    total, shown_loss = 0.0, 0.0
    for j, (inst, spec) in enumerate(chosen):
        a, b = merged.node_segments[j]
        hist_j = [{"p_g": h["p_g"][a:b], "vm_set": h["vm_set"][a:b]} for h in hist_m]
        loss, n_blocks = loss_from_history(hist_j, inst, spec)
        total = total + loss / (n_blocks * BATCH)
        shown_loss += float(loss.detach()) / n_blocks
    return total, shown_loss / BATCH


@torch.no_grad()
def val_probe(model, pools: list, n_max: int) -> dict:
    """Progress signal on the val split: decisions through the terminal PV solve.
    The reference bus's active power comes from the solve, never from the model."""
    per_grid = {}
    for pool in pools:
        vm_mape, th_mae, pg_mae, cost_mape, n_failed = [], [], [], [], 0
        for inst, spec, lab in list(zip(pool.insts, pool.specs, pool.labels))[:n_max]:
            out = model(inst, spec)
            vm_set = torch.where(inst.m_free, out["vm_set"],
                                 torch.ones_like(out["vm_set"]))
            pv = inst.m_free & ~inst.tgrid.is_slack
            pf = transmission_pf_newton_pv(inst.tgrid, out["p_g"], vm_set, pv,
                                           n_iter=40)
            if float(pf["residual"]) > 1e-6:
                n_failed += 1
                continue
            V = pf["V"]
            vm_mape.append(float(((V.abs() - inst.vm_ref).abs()
                                  / inst.vm_ref).mean()) * 100)
            th_mae.append(float(np.rad2deg(
                float((torch.angle(V) - inst.va_ref).abs().mean()))))
            m = inst.m_node_ns
            if int(m.sum()):
                pg_mae.append(float((out["p_g"][m] - inst.pg_label[m]).abs().mean()))
            p_final = out["p_g"].detach() + pf["p_slack"].detach().to(torch.float64)
            pg_gen = allocate_bus_p_exact(p_final, lab, inst.n_nodes)
            cost = float(gen_cost(pg_gen, lab.c2, lab.c1, lab.c0))
            cost_mape.append(abs(cost - lab.cost) / max(abs(lab.cost), 1.0))
        per_grid[pool.alias] = {
            "cost_mape_pct": (float(np.mean(cost_mape)) * 100.0
                              if cost_mape else None),
            "v_mape": float(np.mean(vm_mape)) if vm_mape else None,
            "th_mae_deg": float(np.mean(th_mae)) if th_mae else None,
            "pg_mae": float(np.mean(pg_mae)) if pg_mae else None,
            "n_nonconverged": n_failed,
            "n_eval": min(n_max, len(pool)),
        }
    return per_grid


def load_pools(args, n_cost_feats: int, device) -> tuple:
    """The train and val pools of every training grid, checked against each other
    and against OPFData's own objective."""
    train_pools = [load_pool(g, VARIANT, "train", n=args.n_train,
                             n_cost_feats=n_cost_feats, device=device)
                   for g in args.train]
    val_pools = [load_pool(g, VARIANT, "val", n=args.n_val,
                           n_cost_feats=n_cost_feats, device=device)
                 for g in args.train]
    assert_disjoint(train_pools, val_pools)
    for pool in train_pools:
        check = label_cross_check(pool)
        if not check["pass"]:
            raise SystemExit(f"label cost cross-check failed on {pool.alias}: "
                             f"rel_max={check['rel_max']:.3e}")
    return train_pools, val_pools


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train an OPF generalist.")
    ap.add_argument("--maps", default="physics", choices=["physics", "generic"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ed-prior", action="store_true",
                    help="add the two economic-dispatch columns to the input")
    ap.add_argument("--train", default=",".join(TRAIN_GRIDS),
                    help="comma-separated training grids")
    ap.add_argument("--steps", type=int, default=20000,
                    help=f"optimizer steps; scenario visits = steps x {BATCH}")
    ap.add_argument("--n-train", type=int, default=8000,
                    help="training scenarios per grid, evenly subsampled")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="continue from this run's snapshot in --out-dir")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing a finished record")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.train = [g.strip() for g in args.train.split(",") if g.strip()]
    return args


def train(model, opt, sched, rng, train_pools: list, args, snapshot: Path) -> dict:
    """The loop; returns the state a snapshot keeps, without the weights."""
    pool_index = [(gi, si) for gi, p in enumerate(train_pools)
                  for si in range(len(p))]
    log_every = max(1, args.steps // 20)
    start, history, spent = 0, [], 0.0
    if args.resume and snapshot.exists():
        blob = torch.load(snapshot, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        rng.bit_generator.state = blob["rng_state"]
        start, history, spent = blob["step"], blob["history"], blob["train_s"]
        print(f"resumed from step {start}", flush=True)
    t0 = time.time() - spent
    for step in range(start, args.steps):
        opt.zero_grad()
        picks = rng.choice(len(pool_index), size=BATCH, replace=False)
        chosen = [(train_pools[gi].insts[si], train_pools[gi].specs[si])
                  for gi, si in (pool_index[pi] for pi in picks)]
        total, shown_loss = batch_loss(model, chosen)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            history.append({"step": step + 1, "loss": shown_loss})
            print(f"  step {step + 1}/{args.steps}: loss={shown_loss:.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % SNAPSHOT_EVERY == 0 and (step + 1) < args.steps:
            save_snapshot(snapshot, {
                "step": step + 1, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_state": rng.bit_generator.state, "history": history,
                "train_s": time.time() - t0})
    return {"history": history, "train_s": time.time() - t0, "n_pool": len(pool_index)}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    n_cost_feats = 12 if args.ed_prior else 10
    stem = f"opf_generalist_{args.maps}{'_ed' if args.ed_prior else ''}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{stem}.json").exists() and not args.overwrite:
        raise SystemExit(f"{shown(out_dir / stem)}.json exists; pass --overwrite "
                         "to replace it")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_pools, val_pools = load_pools(args, n_cost_feats, device)
    model = PowerForm(d_model=D_MODEL, K=K, n_cost_feats=n_cost_feats, maps=args.maps,
                      generic_hidden=GENERIC_HIDDEN[args.maps]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{stem}: {params:,} parameters on {device}, {args.steps} steps",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=1e-5)
    snapshot = out_dir / f"{stem}.resume.pt"
    run = train(model, opt, sched, rng, train_pools, args, snapshot)

    model.eval()
    probe = val_probe(model, val_pools, args.n_val)
    for grid, m in probe.items():
        print(f"  val {grid}: cost {m['cost_mape_pct']}%  "
              f"nonconverged {m['n_nonconverged']}/{m['n_eval']}", flush=True)
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "maps": args.maps, "seed": args.seed, "d_model": D_MODEL, "K": K,
        "n_cost_feats": n_cost_feats, "generic_hidden": GENERIC_HIDDEN[args.maps],
        "train_grids": args.train, "variant": VARIANT,
        "n_train_per_grid": args.n_train, "n_train_total": run["n_pool"],
        "steps": args.steps, "batch": BATCH, "scenario_visits": args.steps * BATCH,
        "lr": LR, "clip": CLIP, "w_bal": W_BAL, "params": params,
        "train_s": run["train_s"],
        "data_provenance": {"train": [p.provenance() for p in train_pools],
                            "val": [p.provenance() for p in val_pools]},
        "val_probe": probe, "loss_history": run["history"]}, indent=2))
    torch.save({"model": model.state_dict(), "config": config_of(model)},
               out_dir / f"{stem}.pt")
    snapshot.unlink(missing_ok=True)
    print(f"-> {shown(out_dir / stem)}.json/.pt", flush=True)


if __name__ == "__main__":
    main()
