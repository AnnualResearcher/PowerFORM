"""Table 1 rows at a larger budget, for the saturation runs of App. D (Figure 2).

The loops are those of `powerform/training/case500.py` (physics maps) and
`powerform/model/baselines/recipe.py` (GridSFM decision), unchanged but for the step
count, so the cosine schedule ends at the floor at the new budget. They save only at
the end; this module adds, around the same functions:

  * a snapshot every `--snap-every` steps, from which a killed run continues exactly
    when the same command is started again with `--resume`;
  * the model alone every `--milestone-every` steps, scored later by
    `powerform.eval.milestones` for the training curve;
  * the mean training loss and the learning rate per `--log-every` steps.

Nothing here reads the hold-out during training. Each budget writes to its own folder,
`outputs/long_run/steps<N>/` by default; `powerform.experiments.long_run_records` turns a
finished run into the record files Figure 2 reads.

    python -m powerform.training.long_run --arch physics --seed 0 --steps 120000
    python -m powerform.training.long_run --arch gridsfm --seed 0 --steps 120000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from powerform.data.datakit_loader import (
    TransmissionInstance, build_static, corpus_ids, load_case500, pyg_split, raw_dir_of,
    scenario_labels, table1_split, template_gen_df,
)
from powerform.device import resolve_device
from powerform.eval.checkpoint import config_of
from powerform.model.baselines import recipe
from powerform.model.baselines.common import default_dtype, spread
from powerform.paths import OUTPUTS, shown
from powerform.training import case500
from powerform.training.snapshot import save_snapshot

OUT_DIR = OUTPUTS / "long_run"
# Settings a resumed run must share with the snapshot it continues.
RESUME_MUST_MATCH = ("arch", "seed", "steps", "n_train", "snap_every", "milestone_every",
                     "log_every")


def work_dir(steps: int) -> Path:
    """The default folder of one budget's runs."""
    return OUT_DIR / f"steps{steps}"


class Progress:
    """Snapshots, milestones and the loss log of one run."""

    def __init__(self, args, extra: argparse.Namespace, model: torch.nn.Module,
                 opt: torch.optim.Optimizer, sched, rng: np.random.Generator, config: dict) -> None:
        stem = args.ckpt_path.stem
        self.extra, self.model, self.opt, self.sched, self.rng = extra, model, opt, sched, rng
        self.config = config
        self.settings = {k: getattr(extra, k) for k in RESUME_MUST_MATCH}
        self.snap = Path(extra.work) / "snapshots" / f"{stem}.snap.pt"
        self.miles = Path(extra.work) / "milestones" / stem
        self.curve = Path(extra.work) / "curves" / f"{stem}.jsonl"
        for d in (self.snap.parent, self.miles, self.curve.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.start, self.elapsed, self.restarts = 0, 0.0, 0
        self.sum, self.n = 0.0, 0
        if self.snap.exists() and not extra.resume:
            if not extra.overwrite:
                raise SystemExit(f"{shown(self.snap)} exists; pass --resume to continue that run "
                                 "or --overwrite to start over")
            self.snap.unlink()
        if not self.snap.exists():
            self.curve.unlink(missing_ok=True)      # a fresh start: one run per log
        else:
            # A snapshot holds numpy's RNG state, so it is not a weights-only file.
            s = torch.load(self.snap, map_location="cpu", weights_only=False)
            if s.get("settings") != self.settings:
                raise SystemExit(f"{shown(self.snap)} was written with {s.get('settings')}, "
                                 f"not {self.settings}")
            model.load_state_dict(s["model"])
            opt.load_state_dict(s["opt"])
            sched.load_state_dict(s["sched"])
            rng.bit_generator.state = s["rng"]
            torch.set_rng_state(s["torch_rng"])
            if torch.cuda.is_available() and s.get("cuda_rng"):
                torch.cuda.set_rng_state_all(s["cuda_rng"])
            self.start, self.elapsed = int(s["step"]), float(s["elapsed"])
            self.restarts = int(s["restarts"]) + 1
            self._trim_curve()
            print(f"resumed from step {self.start} ({self.restarts} restart(s))", flush=True)
        self.t0 = time.time()

    def _trim_curve(self) -> None:
        """Drop the log rows written after the snapshot, which the resumed run writes again."""
        if self.curve.exists():
            rows = [line for line in self.curve.read_text().splitlines()
                    if line.strip() and json.loads(line)["step"] <= self.start]
            self.curve.write_text("".join(f"{r}\n" for r in rows))

    def seconds(self) -> float:
        return self.elapsed + time.time() - self.t0

    def after_step(self, step: int, loss: float) -> None:
        self.sum += loss
        self.n += 1
        e = self.extra
        if step % e.log_every == 0:
            row = {"step": step, "loss_mean": self.sum / self.n, "n": self.n,
                   "lr": self.sched.get_last_lr()[0], "s": round(self.seconds(), 1)}
            with self.curve.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  step {step}/{e.steps}: mean loss {row['loss_mean']:.5f} "
                  f"lr {row['lr']:.2e} ({row['s']:.0f}s)", flush=True)
            self.sum, self.n = 0.0, 0
        if step % e.milestone_every == 0:
            torch.save({"model": self.model.state_dict(), "config": self.config, "step": step},
                       self.miles / f"step{step:06d}.pt")
        stopping = e.stop_after and step == e.stop_after
        if step % e.snap_every == 0 or stopping:
            save_snapshot(self.snap, {
                "step": step, "elapsed": self.seconds(), "restarts": self.restarts,
                "settings": self.settings,
                "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
        if stopping:
            print(f"stopping at step {step} on request", flush=True)
            raise SystemExit(3)

    def finish(self) -> None:
        self.snap.unlink(missing_ok=True)


def train_physics(args, extra) -> None:
    """`case500.train_powerform`, with `Progress` added."""
    device = resolve_device(args.device)
    model = case500.init_powerform(args)
    with default_dtype(torch.float64):
        train_ids, hold_ids = table1_split(corpus_ids(args.data_dir))
        train_ids = spread(train_ids, args.n_train)
        print(f"loading {shown(args.data_dir)} ({len(train_ids)} train / "
              f"{len(hold_ids)} hold) ...", flush=True)
        data = load_case500(args.data_dir, scenario_ids=(
            train_ids + hold_ids if args.n_train else None))
        if data.scenario_ids != train_ids + hold_ids:
            raise SystemExit("loaded scenarios are not the Table 1 split")
        train_s = data.scenarios[:len(train_ids)]
        hold_s = data.scenarios[len(train_ids):]
        gen_df = template_gen_df(args.data_dir)
        spec, m_free = case500.decision_spec(data, device)

        model = model.to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"{args.ckpt_path.stem}: {n_par:,} parameters", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=recipe.LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=recipe.ETA_MIN)
        rng = np.random.default_rng(args.seed)
        pg_idx = torch.where(data.m_node)[0].to(device)
        vm_idx = torch.where(m_free)[0].to(device)
        final_only = args.loss == "final-only"

        prog = Progress(args, extra, model, opt, sched, rng, config_of(model))
        for ep in range(prog.start, args.steps):
            opt.zero_grad()
            batch = rng.choice(len(train_s), size=recipe.BATCH, replace=False)
            insts = [TransmissionInstance(data.grid, train_s[bi], device=device)
                     for bi in batch]
            loss_val = case500.step_loss(model, insts, spec, pg_idx, vm_idx, final_only)
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.GRAD_CLIP)
            opt.step()
            sched.step()
            prog.after_step(ep + 1, loss_val / recipe.BATCH)
        train_time = prog.seconds()

        print("evaluating ...", flush=True)
        model.eval()
        res = case500.evaluate(model, data, spec, m_free, hold_s, gen_df, device)
    recipe.save_checkpoint(args.ckpt_path, {"model": model.state_dict(),
                                            "config": config_of(model)})
    recipe.save_record(args.record_path, {
        "seed": args.seed, "n_train": len(train_s), "steps": args.steps,
        "samples": args.steps * recipe.BATCH, "params": n_par, "train_s": train_time,
        "restarts": prog.restarts, **res})
    prog.finish()


def train_gridsfm(args, extra) -> None:
    """`recipe.train_decision` for `--arch gridsfm`, with `Progress` added."""
    from powerform.model.baselines.gridsfm import GraphPrep, decision_outputs

    device = resolve_device(args.device)
    seed = args.seed
    raw = raw_dir_of(args.data_dir)
    st = build_static(raw)
    m_node = st["m_node"].to(device)
    m_free = st["m_free"].to(device)
    train_paths, hold_paths = pyg_split(args.pyg_dir)
    train_paths = spread(train_paths, args.n_train)
    print(f"device={device}  train={len(train_paths)}  holdout="
          f"{len(hold_paths)}  budget={args.steps} steps x batch {recipe.BATCH}",
          flush=True)

    hold_l = [scenario_labels(p, st) for p in hold_paths]
    config = recipe.decision_config(args.arch)
    prep = GraphPrep()
    train_l = [scenario_labels(p, st) for p in train_paths]
    torch.manual_seed(seed)
    model = recipe.build_gridsfm_decision().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_par:,} parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=recipe.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=recipe.ETA_MIN)
    rng = np.random.default_rng(seed)
    model.train()
    prog = Progress(args, extra, model, opt, sched, rng, config)
    for step in range(prog.start, args.steps):
        sel = rng.choice(len(train_paths), size=recipe.BATCH, replace=False)
        pgb, vset = decision_outputs(
            model, [prep(train_paths[i]) for i in sel], st, device)
        pg_l = torch.stack([train_l[i]["pg_label"] for i in sel]
                           ).to(device).to(pgb.dtype)
        vm_l = torch.stack([train_l[i]["vm_ref"] for i in sel]
                           ).to(device).to(vset.dtype)
        loss = recipe.decision_loss(pgb, vset, pg_l, vm_l, m_node, m_free, recipe.BATCH)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.GRAD_CLIP)
        opt.step()
        sched.step()
        prog.after_step(step + 1, float(loss))
    train_time = prog.seconds()

    print("evaluating on the hold-out through the terminal power flow ...", flush=True)
    model.eval()
    decide = recipe.gridsfm_decider(model, hold_paths, st, device)
    res = recipe.score_decisions(decide, hold_l, st)
    recipe.save_checkpoint(args.ckpt_path, {"model": model.state_dict(), "config": config})
    recipe.save_record(args.record_path, {
        "seed": seed, "n_train": len(train_paths), "steps": args.steps,
        "samples": args.steps * recipe.BATCH, "params": n_par, "train_s": train_time,
        "restarts": prog.restarts, **res})
    prog.finish()


def parse(argv: Optional[List[str]] = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """(the Table 1 row's arguments, this module's own), from one command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arch", required=True, choices=("physics", "gridsfm"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--work", default=None, help="default outputs/long_run/steps<N>")
    ap.add_argument("--resume", action="store_true",
                    help="continue from this run's snapshot in --work")
    ap.add_argument("--overwrite", action="store_true",
                    help="start over, replacing a snapshot, record or checkpoint of this run")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=None, help="as for powerform.training.case500")
    ap.add_argument("--pyg-dir", default=None, help="as for powerform.training.case500")
    ap.add_argument("--snap-every", type=int, default=500)
    ap.add_argument("--milestone-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--stop-after", type=int, default=0,
                    help="exit after this step with a snapshot, to test resuming")
    extra = ap.parse_args(argv)
    if extra.steps % extra.milestone_every:
        ap.error("--steps must be a multiple of --milestone-every, so the last milestone is the run's end")
    extra.work = str(work_dir(extra.steps)) if extra.work is None else extra.work
    out = Path(extra.work) / "out"
    row_args = ["--arch", extra.arch, "--seed", str(extra.seed), "--steps", str(extra.steps),
                "--n-train", str(extra.n_train), "--out-dir", str(out), "--device", extra.device]
    for flag, value in (("--data-dir", extra.data_dir), ("--pyg-dir", extra.pyg_dir)):
        if value is not None:
            row_args += [flag, value]
    args = case500.parse_args(row_args)
    return args, extra


def main(argv: Optional[List[str]] = None) -> None:
    args, extra = parse(argv)
    case500.refuse_overwrite(args.record_path, extra.overwrite)
    case500.refuse_overwrite(args.ckpt_path, extra.overwrite)
    if extra.arch == "physics":
        train_physics(args, extra)
    else:
        train_gridsfm(args, extra)


if __name__ == "__main__":
    main()
