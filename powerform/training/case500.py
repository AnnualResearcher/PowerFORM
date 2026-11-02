"""In-distribution OPF on case500_goc: trains and re-scores every Table 1 row.

Table 1 compares backbones and training objectives on one corpus
(`data/datakit_loader.py`) under one budget, 20,000 steps of batch 4. The App. D
in-distribution table removes one design element of the generic block per row.

    --arch physics                      PowerFORM
    --arch generic                      PowerFORM, generic attention
    --arch generic --no-type-masking    F1, no type masking
    --arch generic --merged-readout     F2, merged readout
    --arch generic --loss final-only    F3, final-layer supervision
    --arch mlp | gnn                    decision
    --arch gridsfm [--recipe state]     decision, or state regression
    --arch graphkit                     decision, in graphkit's environment

The graphkit state-regression row runs graphkit's own trainer
(`model/baselines/README.md`).

Decision recipe, PowerForm rows: the model emits active dispatch at the non-slack
generator buses and a voltage setpoint at every free bus. Every block's readout is
supervised against the straight path from the flat start (p_g = 0, V = 1) to the
IPOPT label, stage t_k = (k+1)/K at block k, plus a total-generation term. Reactive
dispatch, angles and flows come out of the terminal PV power flow at evaluation. The
baselines' loop is in `model/baselines/recipe.py`.

`--eval-only CKPT` re-scores a trained checkpoint on the 500 hold-out scenarios and
reads what the row is from the checkpoint's own config. A graphkit `predict` output
directory stands in for the checkpoint of the graphkit state-regression row.

A run writes `<out-dir>/<row>_seed<s>.{pt,json}`, and a re-score
`<out-dir>/rescore/<row>_seed<s>.json`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from powerform.data.batch import merge_instances
from powerform.data.corpus.datakit_export import CORPUS_DIR, DATAKIT_HOME
from powerform.data.datakit_loader import (
    Case500Data, Scenario, TransmissionInstance, corpus_ids, dispatch_cost,
    load_case500, slack_quadratic, table1_split, template_gen_df,
)
from powerform.device import resolve_device
from powerform.eval.checkpoint import config_of, load_model, read_checkpoint
from powerform.model.baselines.common import (
    default_dtype, report_progress, spread,
)
from powerform.model.baselines.recipe import (
    BATCH, ETA_MIN, GRAD_CLIP, LR, PF_ITERS, PF_TOL, VSET_CLAMP, VSET_WEIGHT, W_BAL,
    rescore_baseline, save_checkpoint, save_record, train_baseline,
)
from powerform.model.powerform import PowerForm
from powerform.model.spec import TaskSpec
from powerform.paths import OUTPUTS, shown
from powerform.physics.newton_solver import transmission_pf_newton_pv

D_MODEL = 216
K = 8
STEPS = 20000
MAPS = ("physics", "generic")
BASELINES = ("mlp", "gnn", "gridsfm", "graphkit")

# Generic-block MLP width: 450 keeps the generic arms within 0.5 % of the physics
# maps' parameter count.
GENERIC_HIDDEN = 450


# ---------------------------------------------------------------------------
# Task binding and model construction
# ---------------------------------------------------------------------------
def decision_spec(data: Case500Data, device) -> tuple:
    """The decision task: free buses = non-slack generator buses + slack.

    Returns `(spec, m_free)`. The p box is the per-bus sum of non-slack unit boxes
    with the slack zeroed, since the slack injection is a consequence of the power
    flow and never a decision.
    """
    n = data.grid.n_nodes
    slack_mask = data.grid.is_slack
    m_free = data.m_node | slack_mask
    p_min = torch.where(slack_mask, torch.zeros(n, dtype=torch.float64),
                        data.p_min)
    p_max = torch.where(slack_mask, torch.zeros(n, dtype=torch.float64),
                        data.p_max)
    spec = TaskSpec(m_node=m_free.to(device),
                    m_edge=torch.zeros(data.grid.n_edges, dtype=torch.bool,
                                       device=device),
                    p_min=p_min.to(device), p_max=p_max.to(device),
                    vm_min=data.vm_min.to(device),
                    vm_max=data.vm_max.to(device))
    return spec, m_free


def init_powerform(args) -> PowerForm:
    """The untrained model a run starts from.

    Weights are initialized in float64: initializing in float32 and casting would
    draw different weights from the same seed. Loading the corpus draws nothing
    from torch's generator, so the model can be built before the data.
    """
    with default_dtype(torch.float64):
        torch.manual_seed(args.seed)
        return PowerForm(d_model=D_MODEL, K=K, maps=args.arch,
                         generic_hidden=(GENERIC_HIDDEN if args.arch == "generic"
                                         else None),
                         type_masking=not args.no_type_masking,
                         merged_readout=args.merged_readout)


# ---------------------------------------------------------------------------
# The layerwise decision loss
# ---------------------------------------------------------------------------
def instance_loss(hist: Sequence[Dict], inst, pg_idx: torch.Tensor,
                  vm_idx: torch.Tensor, final_only: bool) -> torch.Tensor:
    """Sum over blocks of the interpolant loss for one instance.

    Block k targets t_k = (k+1)/K of the way from the flat start to the label:
    p_g -> t_k * p_label and vm -> 1 + t_k (vm_label - 1), plus the squared error
    of total generation weighted by `W_BAL / |gen buses|`. `final_only` keeps the
    last block's term; the caller still divides by K.
    """
    KK = len(hist)
    pg_l = inst.pg_label[pg_idx]
    vm_l = inst.vm_ref[vm_idx]
    inst_loss = 0.0
    for k, S_k in enumerate(hist):
        if final_only and k != KK - 1:
            continue
        t_k = (k + 1) / KK
        sup_p = ((S_k["p_g"][pg_idx] - t_k * pg_l) ** 2).sum()
        vm_t = 1.0 + t_k * (vm_l - 1.0)
        sup_v = ((S_k["vm_set"][vm_idx] - vm_t) ** 2).sum() * VSET_WEIGHT
        bal = (S_k["p_g"][pg_idx].sum() - t_k * pg_l.sum()) ** 2
        inst_loss = inst_loss + (sup_p + sup_v
                                 + W_BAL * bal / len(pg_idx))
    return inst_loss


def step_loss(model: PowerForm, insts: list, spec: TaskSpec,
              pg_idx: torch.Tensor, vm_idx: torch.Tensor,
              final_only: bool) -> float:
    """One optimizer step's backward pass; returns the logging value.

    The instances run as one disjoint graph (`data/batch.py`), and the merged
    history is sliced back per instance. The logged value is the per-instance loss
    averaged over blocks, summed over instances.
    """
    merged, mspec = merge_instances(insts, [spec] * len(insts))
    hist_m = model(merged, mspec)["history"]
    hists = [[{"p_g": h["p_g"][a:b], "vm_set": h["vm_set"][a:b]} for h in hist_m]
             for a, b in merged.node_segments]
    n_b = len(insts)
    total = 0.0
    loss_val = 0.0
    for inst, hist in zip(insts, hists):
        KK = len(hist)
        inst_loss = instance_loss(hist, inst, pg_idx, vm_idx, final_only)
        total = total + inst_loss / (KK * n_b)
        loss_val += float(inst_loss.detach()) / KK
    total.backward()
    return loss_val


# ---------------------------------------------------------------------------
# Evaluation through the terminal power flow
# ---------------------------------------------------------------------------
def evaluate(model: PowerForm, data: Case500Data, spec: TaskSpec,
             m_free: torch.Tensor, hold: Sequence[Scenario], gen_df,
             device) -> Dict:
    """Score `model` on `hold` through the PV power flow.

    Decodes (p_g, vm_set), clamps the setpoints to `VSET_CLAMP`, solves, and scores
    the solved state against the labels. Returns the means of V MAPE (%), theta MAE
    (deg), P_g MAE and Q_g MAE (pu, bus level, non-slack generator buses) and cost
    MAPE (%), the non-converged count and the per-scenario cost MAPE. Cost allocates
    each bus's dispatch to its units at equal marginal cost and adds the slack's
    cost at the solved slack injection.
    """
    s_coef = slack_quadratic(gen_df)
    ch = {"v_mape": [], "th_mae": [], "pg_mae": [], "qg_mae": [],
          "cost_mape": []}
    n_nonconv = 0
    gen_bus_np = data.gen_bus.numpy()
    g0 = gen_df[gen_df.in_service > 0]
    ns = g0[~g0.is_slack_gen.astype(bool)]
    t0 = time.time()
    with torch.no_grad():
        for i, scen in enumerate(hold):
            inst = TransmissionInstance(data.grid, scen, device=device)
            out = model(inst, spec)
            vm_set = torch.where(m_free.to(device),
                                 out["vm_set"].clamp(*VSET_CLAMP),
                                 torch.ones_like(out["vm_set"]))
            pv = m_free.to(device) & ~inst.tgrid.is_slack
            pf = transmission_pf_newton_pv(
                inst.tgrid, out["p_g"], vm_set, pv, n_iter=PF_ITERS)
            if float(pf["residual"]) > PF_TOL:
                n_nonconv += 1
            V = pf["V"]
            vm, va = V.abs(), torch.angle(V)
            ch["v_mape"].append(float(
                ((vm - inst.vm_ref).abs() / inst.vm_ref).mean()) * 100)
            ch["th_mae"].append(float(np.rad2deg(float(
                (va - inst.va_ref).abs().mean()))))
            m = data.m_node.to(device)
            ch["pg_mae"].append(float(
                (out["p_g"][m] - inst.pg_label[m]).abs().mean()))
            ch["qg_mae"].append(float(
                (pf["q_g"][m] - inst.qg_label[m]).abs().mean()))
            cost = dispatch_cost(out["p_g"].cpu().numpy(),
                                 float(pf["p_slack"].sum()), gen_bus_np, ns,
                                 s_coef)
            ch["cost_mape"].append(
                abs(cost - scen.cost_ref) / max(scen.cost_ref, 1e-9) * 100)
            report_progress(i + 1, len(hold), t0)
    res = {k: float(np.mean(v)) for k, v in ch.items()}
    res.update({"n_nonconverged": n_nonconv,
                "per_scenario_cost_mape": ch["cost_mape"]})
    return res


# ---------------------------------------------------------------------------
# Training (PowerForm rows)
# ---------------------------------------------------------------------------
def train_powerform(args) -> None:
    device = resolve_device(args.device)
    model = init_powerform(args)
    # Labels and power flow are float64 like the weights.
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
        spec, m_free = decision_spec(data, device)

        model = model.to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"{args.ckpt_path.stem}: {n_par:,} parameters", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=ETA_MIN)
        rng = np.random.default_rng(args.seed)
        pg_idx = torch.where(data.m_node)[0].to(device)
        vm_idx = torch.where(m_free)[0].to(device)
        final_only = args.loss == "final-only"

        t0 = time.time()
        for ep in range(args.steps):
            opt.zero_grad()
            batch = rng.choice(len(train_s), size=BATCH, replace=False)
            insts = [TransmissionInstance(data.grid, train_s[bi], device=device)
                     for bi in batch]
            loss_val = step_loss(model, insts, spec, pg_idx, vm_idx, final_only)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            if (ep + 1) % max(1, args.steps // 10) == 0:
                print(f"  step {ep+1}/{args.steps}: loss={loss_val/BATCH:.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        train_time = time.time() - t0

        print("evaluating ...", flush=True)
        model.eval()
        res = evaluate(model, data, spec, m_free, hold_s, gen_df, device)
    save_checkpoint(args.ckpt_path, {"model": model.state_dict(),
                                     "config": config_of(model)})
    save_record(args.record_path, {
        "seed": args.seed, "n_train": len(train_s),
        "samples": args.steps * BATCH, "params": n_par, "train_s": train_time,
        **res})


# ---------------------------------------------------------------------------
# Re-scoring a trained checkpoint
# ---------------------------------------------------------------------------
def rescore_powerform(ckpt: Path, data_dir, device,
                      limit: Optional[int] = None) -> Dict:
    """Re-score one PowerForm checkpoint on the Table 1 hold-out, in float64."""
    with default_dtype(torch.float64):
        _, hold_ids = table1_split(corpus_ids(data_dir))
        data = load_case500(data_dir, scenario_ids=hold_ids[:limit]
                            if limit else hold_ids)
        gen_df = template_gen_df(data_dir)
        spec, m_free = decision_spec(data, device)
        model = load_model(ckpt, device)
        return evaluate(model, data, spec, m_free, data.scenarios, gen_df, device)


def row_of_checkpoint(ckpt: Path) -> Tuple[str, str]:
    """(arch, recipe) of a trained artifact, read from its own config."""
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} does not exist; the Table 1 checkpoints come from "
                         "python scripts/download_checkpoints.py --group table1")
    if ckpt.is_dir():
        return "graphkit", "state"
    config = read_checkpoint(ckpt)["config"]
    if "maps" in config:
        if config["n_cost_feats"] != 0:
            raise SystemExit(f"{ckpt.name} was not trained on the case500_goc corpus; "
                             "powerform.eval.opf_eval scores the generalists")
        return config["maps"], "decision"
    return config["arch"], config["recipe"]


def rescore(ckpt: Path, args, device, limit: Optional[int] = None) -> Dict:
    """Re-score one trained artifact as the row its config says it is."""
    arch, recipe = row_of_checkpoint(ckpt)
    if args.arch not in (None, arch):
        raise SystemExit(f"{ckpt.name} is an --arch {arch} checkpoint, "
                         f"not --arch {args.arch}")
    print(f"re-scoring {ckpt.name}: {arch}, {recipe}", flush=True)
    if arch in MAPS:
        return rescore_powerform(ckpt, args.data_dir, device, limit=limit)
    return rescore_baseline(arch, recipe, ckpt, args, device, limit=limit)


def refuse_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{shown(path)} exists; pass --overwrite to replace it")


def eval_only(args) -> None:
    device = resolve_device(args.device)
    limit = args.limit or None
    for ck in map(Path, args.eval_only):
        out = Path(args.out_dir) / "rescore" / (
            ck.stem + (f"_first{limit}" if limit else "") + ".json")
        refuse_overwrite(out, args.overwrite)
        save_record(out, rescore(ck, args, device, limit=limit))


# ---------------------------------------------------------------------------
def row_name(args) -> str:
    """The row a training run belongs to, which names its checkpoint and record."""
    if args.arch == "generic":
        return "_".join(["case500_generic"]
                        + ["no_type_masking"] * args.no_type_masking
                        + ["merged_readout"] * args.merged_readout
                        + ["final_layer_only"] * (args.loss == "final-only"))
    if args.arch in ("gridsfm", "graphkit"):
        return f"case500_{args.arch}_{args.recipe}"
    return f"case500_{args.arch}"


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        description="Train or re-score a Table 1 / App. D row on the "
                    "case500_goc datakit corpus.")
    ap.add_argument("--arch", default=None, choices=MAPS + BASELINES,
                    help="default physics; a re-score reads it from the "
                         "checkpoint")
    ap.add_argument("--recipe", default="decision",
                    choices=("decision", "state"),
                    help="state = full-state regression, for gridsfm and "
                         "graphkit; a re-score reads it from the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--n-train", type=int, default=0,
                    help="n training scenarios spread evenly over the 29,487 "
                         "(0 = all); the GridSFM state row takes the first n")
    ap.add_argument("--out-dir", default=str(OUTPUTS / "case500"))
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a finished record of the same row and seed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=str(CORPUS_DIR),
                    help="datakit export root, holding case500_goc/raw")
    ap.add_argument("--pyg-dir", default=str(DATAKIT_HOME),
                    help="root of the .pyg.json export (train_opfdata/, "
                         "holdout_opfdata/); baseline rows")
    ap.add_argument("--no-type-masking", action="store_true")
    ap.add_argument("--merged-readout", action="store_true")
    ap.add_argument("--loss", default="layerwise",
                    choices=("layerwise", "final-only"))
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers, GridSFM state row")
    ap.add_argument("--graphkit-splits", default=None,
                    help="graphkit split files (graphkit.make_splits)")
    ap.add_argument("--graphkit-norm-stats", default=None,
                    help="the normalizer checkpoint (otherwise refitted on the "
                         "train split)")
    ap.add_argument("--eval-only", nargs="+", default=None, metavar="CKPT",
                    help="re-score these checkpoints instead of training")
    ap.add_argument("--limit", type=int, default=0,
                    help="re-score the first n hold-out scenarios only")
    args = ap.parse_args(argv)
    if args.eval_only:
        return args
    args.arch = args.arch or "physics"
    removed = (args.no_type_masking or args.merged_readout
               or args.loss != "layerwise")
    if removed and args.arch != "generic":
        ap.error("--no-type-masking, --merged-readout and --loss final-only "
                 "apply to --arch generic")
    if args.recipe == "state" and args.arch not in ("gridsfm", "graphkit"):
        ap.error("--recipe state applies to --arch gridsfm and graphkit")
    if args.arch == "graphkit" and args.recipe == "state":
        ap.error("the graphkit state row is graphkit's own trainer: "
                 "python -m powerform.model.baselines.graphkit train "
                 "(see model/baselines/README.md)")
    stem = f"{row_name(args)}_seed{args.seed}"
    args.ckpt_path = Path(args.out_dir) / f"{stem}.pt"
    args.record_path = Path(args.out_dir) / f"{stem}.json"
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.eval_only:
        eval_only(args)
        return
    refuse_overwrite(args.ckpt_path, args.overwrite)
    refuse_overwrite(args.record_path, args.overwrite)
    if args.arch in MAPS:
        train_powerform(args)
    else:
        train_baseline(args)


if __name__ == "__main__":
    main()
