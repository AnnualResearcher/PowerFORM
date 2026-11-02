"""Training the line-opening policy on case57 under N-1 contingencies.

Labels are found online. Each visit to a scenario scores the model's current
opening, Gumbel-perturbed variants and the best opening known for that scenario
with an AC power flow; the best feasible one known so far is the target of a
hardest-pair ranking hinge on the log-gates. Candidates that fail to converge or
leave the voltage box are discarded, so a whole batch can come out infeasible. It
then takes no optimiser step, but still reaches evaluation and snapshots.

    python -m powerform.training.reconfiguration --maps physics --arm pretrained --seed 0
    python -m powerform.training.reconfiguration --maps physics --arm scratch --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from powerform.data.binding import reconfiguration_binding
from powerform.device import resolve_device
from powerform.eval.checkpoint import build_model, config_of
from powerform.eval.scorer import (control_scores, eval_action, has_reference,
                                   score, vm_box_of)
from powerform.experiments.reference import (GRID, K_OPEN, N_COST_FEATS,
                                             N_SCENARIOS, OUT_DIR,
                                             reference_table, task_pools)
from powerform.model.line_track import ReconfigurationNet
from powerform.model.powerform import PowerForm
from powerform.paths import CHECKPOINTS, RESULTS, shown
from powerform.training.generalist import GENERIC_HIDDEN
from powerform.training.snapshot import save_snapshot

D_MODEL, K = 216, 8
EVAL_STEPS = sorted(set(list(range(50, 1001, 50))
                        + list(range(1250, 5001, 250))
                        + list(range(6000, 20001, 1000))))
REFERENCE_NAME = "reference_val.json"
# Settings a resumed run must share with the snapshot it continues.
RESUME_MUST_MATCH = ("maps", "arm", "seed", "steps", "n_train", "batch", "T",
                     "gumbel_tau", "lr", "trunk_lr_mult", "margin", "k_open",
                     "z_track")


class Task(NamedTuple):
    """The pools of a run, on the model's device and on the CPU, and the reference."""
    train_dev: object
    train_cpu: object
    val_dev: object
    val_cpu: object
    ref: dict


def default_init(maps: str, seed: int) -> Path:
    """The generalist an arm starts from: physics seeds 1 and 2 and every generic
    seed from their own, the other physics seeds from generalist seed 0."""
    own = maps == "generic" or seed in (1, 2)
    return CHECKPOINTS / f"opf_generalist_{maps}_ed_seed{seed if own else 0}.pt"


def load_init(args, device) -> dict | None:
    """The generalist checkpoint of a pretrained arm."""
    if args.arm == "scratch":
        return None
    if not Path(args.init).exists():
        raise SystemExit(f"--init {args.init} not found; fetch the generalists "
                         "with: python scripts/download_checkpoints.py --group fig1")
    blob = torch.load(args.init, map_location=device, weights_only=True)
    if blob["config"]["maps"] != args.maps:
        raise SystemExit(f"--init has maps={blob['config']['maps']!r} but this "
                         f"run is --maps {args.maps}")
    return blob


def build_model_for(args, init: dict | None) -> ReconfigurationNet:
    """Both arms build the trunk before the track, which fixes the track's init
    for a given seed; the pretrained arm loads the generalist's weights between
    the two."""
    if init is None:
        trunk = PowerForm(d_model=D_MODEL, K=K, n_cost_feats=N_COST_FEATS,
                          maps=args.maps,
                          generic_hidden=GENERIC_HIDDEN[args.maps])
    else:
        trunk = build_model(init["config"])
        trunk.load_state_dict(init["model"])
    return ReconfigurationNet(trunk, z_d=args.z_track)


def make_optimizer(model, args) -> tuple:
    """AdamW with the backbone at lr x trunk_lr_mult and the track at lr, under a
    cosine schedule. The decision readouts get no gradient here and stay out of
    the optimiser: weight decay would otherwise pull them toward zero."""
    trainable = [p for n, p in model.named_parameters()
                 if "trunk.readouts" not in n]
    track = {id(p) for p in model.trunk.z_stream.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in trainable if id(p) not in track],
          "lr": args.lr * args.trunk_lr_mult},
         {"params": [p for p in trainable if id(p) in track], "lr": args.lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    return trainable, opt, sched


def candidates(logits: torch.Tensor, k: int, T: int, tau: float, rng,
               incumbent: tuple | None) -> list[tuple]:
    """The lowest k lines, T - 1 Gumbel-perturbed draws and the incumbent."""
    lg = logits.detach().double().cpu().numpy()
    out = [tuple(sorted(np.argsort(lg)[:k].tolist()))]
    for _ in range(max(0, T - 1)):
        g = rng.gumbel(size=lg.shape) * tau
        out.append(tuple(sorted(np.argsort(lg + g)[:k].tolist())))
    if incumbent is not None:
        out.append(incumbent)
    # The scoring order decides between equal losses. Both passes through a set
    # shape that order, and the recorded runs depend on it.
    return list(set(list({tuple(sorted(c)) for c in out})))


def ranking_hinge(logits: torch.Tensor, action: tuple, margin: float):
    """The highest log-gate among the lines to open must sit below the lowest
    among the lines to keep by at least ``margin``."""
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[list(action)] = True
    return torch.clamp(margin + logits[mask].max() - logits[~mask].min(), min=0.0)


def scenario_loss(model, train_dev, train_cpu, i: int, buffer: dict, args, rng):
    """Propose, score and clone for one scenario; None if nothing was feasible.

    The target is the best opening known for the scenario, so an unlucky draw
    cannot make the model forget one it has already proved.
    """
    inst2, spec2 = reconfiguration_binding(train_dev.insts[i], train_dev.labels[i],
                                           train_dev.specs[i])
    logits = model(inst2, spec2)
    incumbent = buffer.get(i)
    feasible = []
    for c in candidates(logits, args.k_open, args.T, args.gumbel_tau, rng,
                        incumbent[0] if incumbent else None):
        loss = eval_action(train_cpu.insts[i], train_cpu.labels[i], c,
                           vm_box_of(train_cpu, i))
        if loss is not None:
            feasible.append((c, loss))
    if not feasible:
        return None
    best = min(feasible, key=lambda t: t[1])
    if incumbent is None or best[1] < incumbent[1]:
        buffer[i] = incumbent = best
    return ranking_hinge(logits, incumbent[0], args.margin)


def reference_for(pool_cpu, k: int, outages: list[int], out_dir: Path) -> dict:
    """The validation reference of this task: a copy in ``out_dir``, then the
    shipped one; otherwise it is built, which takes a few minutes."""
    target = out_dir / REFERENCE_NAME
    for path in (target, RESULTS / "reconfiguration" / REFERENCE_NAME):
        if path.exists():
            doc = json.loads(path.read_text())
            if (doc["k"], doc["n"]) == (k, len(pool_cpu.insts)):
                break
    else:
        print(f"building the k={k} reference ...", flush=True)
        doc = reference_table(pool_cpu, "val", k, outages)
    if path != target:
        target.write_text(json.dumps(doc, indent=1))
    return doc


def resume(path: Path, model, opt, sched, rng, settings: dict) -> dict:
    """Restore a run: model, optimiser, schedule, label buffer and sampling RNG."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    bad = {k: (v, blob["settings"].get(k)) for k, v in settings.items()
           if blob["settings"].get(k) != v}
    if bad:
        raise SystemExit(f"{path.name} was written with other settings, so this "
                         f"would not continue that run: {bad}")
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    rng.bit_generator.state = blob["rng_state"]
    return blob


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the line-opening policy.")
    ap.add_argument("--maps", default="physics", choices=["physics", "generic"])
    ap.add_argument("--arm", required=True, choices=["pretrained", "scratch"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default=None,
                    help="generalist checkpoint of the pretrained arm; default "
                         "the one the paper used for this seed")
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
    args = ap.parse_args()
    if args.init is None:
        args.init = str(default_init(args.maps, args.seed))
    return args


def train(model, opt, sched, trainable, task: Task, args, rng, state: dict,
          snapshot: Path) -> None:
    """The training loop. ``state`` holds what a snapshot keeps: the step, the
    label buffer, the metric rows and the time already spent."""
    t0 = time.time() - state["elapsed_s"]
    eval_steps = {s for s in EVAL_STEPS if s <= args.steps} | {args.steps}
    settings = {k: getattr(args, k) for k in RESUME_MUST_MATCH}
    buffer, rows = state["buffer"], state["rows"]
    model.train()
    for step in range(state["step"] + 1, args.steps + 1):
        terms = []
        for _ in range(args.batch):
            i = int(rng.integers(len(task.train_dev.insts)))
            term = scenario_loss(model, task.train_dev, task.train_cpu, i, buffer,
                                 args, rng)
            if term is not None:
                terms.append(term)
        if terms:
            loss = sum(terms) / len(terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
        if step in eval_steps:
            metrics = score(model, task.val_dev, task.val_cpu, task.ref, args.k_open)
            rows += [{"step": step, "metric": m, "value": v}
                     for m, v in metrics.items()]
            shown_loss = float(loss.detach()) if terms else float("nan")
            print(f"  step {step}: loss {shown_loss:.4f} | median gap "
                  f"{metrics['gap_pct_median']:+.3f}% infeasible "
                  f"{metrics['infeasible_fraction']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_snapshot(snapshot, {
                "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "buffer": buffer,
                "rng_state": rng.bit_generator.state, "rows": rows,
                "elapsed_s": time.time() - t0, "settings": settings})
    state["elapsed_s"] = time.time() - t0


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.maps}_{args.arm}_seed{args.seed}"
    dest = out_dir / f"{stem}.json"
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{shown(dest)} exists; pass --overwrite to replace it")

    init = load_init(args, device)
    n_cost_feats = init["config"]["n_cost_feats"] if init else N_COST_FEATS
    print("loading the pools and placing their outages ...", flush=True)
    train_dev, train_cpu, _ = task_pools("train", args.n_train, n_cost_feats, device)
    val_dev, val_cpu, outages = task_pools("val", N_SCENARIOS["val"], n_cost_feats,
                                           device)
    ref = reference_for(val_cpu, args.k_open, outages, out_dir)
    controls = control_scores(ref)
    print(f"controls: all-closed gap {controls['all_closed_gap_pct']:+.3f}%, "
          f"random-{args.k_open} gap {controls['random_k_gap_pct']:+.3f}%", flush=True)

    model = build_model_for(args, init).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{stem}: {params:,} parameters on {device}", flush=True)
    trainable, opt, sched = make_optimizer(model, args)

    rng = np.random.default_rng(args.seed)
    snapshot = out_dir / f"{stem}.resume.pt"
    state = {"step": 0, "buffer": {}, "rows": [], "elapsed_s": 0.0}
    resumed_at = None
    if args.resume and snapshot.exists():
        blob = resume(snapshot, model, opt, sched, rng,
                      {k: getattr(args, k) for k in RESUME_MUST_MATCH})
        state = {k: blob[k] for k in state}
        resumed_at = state["step"]
        print(f"resumed from step {resumed_at}", flush=True)
    train(model, opt, sched, trainable,
          Task(train_dev, train_cpu, val_dev, val_cpu, ref), args, rng, state,
          snapshot)

    dest.write_text(json.dumps({
        "maps": args.maps, "arm": args.arm, "seed": args.seed, "grid": GRID,
        "steps": args.steps, "k_open": args.k_open, "n_train": len(train_dev.insts),
        "n_val": len(val_dev.insts),
        "n_scored": sum(has_reference(r) for r in ref["rows"]), "batch": args.batch,
        "T": args.T, "gumbel_tau": args.gumbel_tau, "margin": args.margin,
        "lr": args.lr, "trunk_lr_mult": args.trunk_lr_mult, "z_track": args.z_track,
        "generic_hidden": model.trunk.generic_hidden,
        "init": None if args.arm == "scratch" else Path(args.init).name,
        "params": params, "resumed_from_step": resumed_at, "elapsed_s": state["elapsed_s"],
        "controls": controls, "rows": state["rows"]}, indent=1))
    # Stored in float32 to halve the file. Rankings and entropies are unaffected;
    # an exact replay of a decision needs the run's float64 state.
    ckpt = out_dir / f"reconfig_{stem}.pt"
    torch.save({"model": {k: (v.float() if v.is_floating_point() else v)
                          for k, v in model.state_dict().items()},
                "config": config_of(model)}, ckpt)
    snapshot.unlink(missing_ok=True)
    print(f"-> {shown(dest)}, {shown(ckpt)}", flush=True)


if __name__ == "__main__":
    main()
