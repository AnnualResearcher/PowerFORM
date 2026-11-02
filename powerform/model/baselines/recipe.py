"""The decision recipe of the baseline rows, their loops and their evaluation.

The baseline rows read their scenarios from the `.pyg.json` export and run in
float32, so they cannot share PowerForm's loop in `training/case500.py` line for
line. They share its recipe and its evaluation, so that the backbone column of
Table 1 compares backbones and not training setups.

Decision recipe (MLP, GNN, GridSFM decision, graphkit decision): one forward on a
batch of 4 scenarios drawn without replacement, decision MSE on the final output

    sup_p = sum((Pg_bus[m_node] - pg_label[m_node])^2)
    sup_v = sum((Vset[m_free] - vm_ref[m_free])^2) * 100
    bal   = sum over the batch of (sum Pg_bus[m_node] - sum pg_label[m_node])^2
    L     = (sup_p + sup_v + 5 * bal / |m_node|) / batch

with AdamW 3e-4, cosine to 1e-5 over 20,000 steps and gradient clip 1.0. This is
PowerForm's objective at t_k = 1: these backbones have no per-block decision state
to supervise along the path.

Evaluation (`score_decisions`): the terminal PV power flow and the cost scorer the
PowerForm rows use, on the 500 hold-out scenarios. Cost includes the slack units.

State regression: GridSFM's fine-tuning loop from random init, scored on the
emitted state (`gridsfm.py`). The graphkit rows run in graphkit's environment
(`graphkit.py`).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from powerform.data.datakit_loader import (
    BASE_MVA, build_static, predicted_cost, pyg_split, raw_dir_of,
    scenario_labels,
)
from powerform.device import resolve_device
from powerform.model.baselines import gnn, mlp
from powerform.model.baselines.common import (
    build_grid_context, default_dtype, report_progress, scenario_channels, spread,
)
from powerform.paths import shown
from powerform.physics.newton_solver import transmission_pf_newton_pv

BATCH = 4
W_BAL = 5.0
LR = 3e-4
ETA_MIN = 1e-5
GRAD_CLIP = 1.0
VSET_WEIGHT = 100.0
# The setpoints are clamped to this band before the terminal solve.
VSET_CLAMP = (0.90, 1.10)
PF_ITERS = 40
PF_TOL = 1e-6


def save_record(path: Path, record: Dict) -> None:
    """A record of numbers and settings. A training run writes it last, after its
    checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {shown(path)}  cost_mape={record['cost_mape']:.6f} over "
          f"{len(record['per_scenario_cost_mape'])} scenarios", flush=True)


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
def build_bus_backbone(config: Dict, ctx: Dict) -> nn.Module:
    """The MLP or GNN `config` describes, built under float32."""
    arch = config["arch"]
    shape = {k: v for k, v in config.items() if k not in ("arch", "recipe")}
    with default_dtype(torch.float32):
        if arch == "mlp":
            return mlp.MLPBackbone(ctx, **shape)
        if arch == "gnn":
            return gnn.GNNBackbone(ctx, **shape)
    raise ValueError(f"not a bus-feature backbone: {arch!r}")


def build_gridsfm_decision() -> nn.Module:
    with default_dtype(torch.float32):
        from powerform.model.baselines.gridsfm import DecisionHeadGridSFM
        return DecisionHeadGridSFM()


def decision_config(arch: str) -> Dict:
    """What a decision-row checkpoint records to rebuild its model."""
    shape = {"mlp": mlp.CONFIG, "gnn": gnn.CONFIG}.get(arch, {})
    return {"arch": arch, "recipe": "decision", **shape}


# ---------------------------------------------------------------------------
# The decision loss (final output only)
# ---------------------------------------------------------------------------
def decision_loss(pgb: torch.Tensor, vset: torch.Tensor, pg_l: torch.Tensor,
                  vm_l: torch.Tensor, m_node: torch.Tensor,
                  m_free: torch.Tensor, batch: int) -> torch.Tensor:
    """The decision recipe at t_k = 1 on a [batch, n_bus] pair of outputs."""
    n_m = int(m_node.sum())
    sup_p = ((pgb[:, m_node] - pg_l[:, m_node]) ** 2).sum()
    sup_v = ((vset[:, m_free] - vm_l[:, m_free]) ** 2).sum() * VSET_WEIGHT
    bal = ((pgb[:, m_node].sum(dim=1)
            - pg_l[:, m_node].sum(dim=1)) ** 2).sum()
    return (sup_p + sup_v + W_BAL * bal / n_m) / batch


# ---------------------------------------------------------------------------
# Corpus views
# ---------------------------------------------------------------------------
def bus_feature_corpus(paths: List[Path], st: Dict) -> Dict:
    """Scenario channels and decision labels of `paths`, as float32 arrays."""
    var = np.empty((len(paths), st["n"], 5), dtype=np.float32)
    pg = np.empty((len(paths), st["n"]), dtype=np.float32)
    vm = np.empty((len(paths), st["n"]), dtype=np.float32)
    for i, p in enumerate(paths):
        lab = scenario_labels(p, st)
        var[i] = scenario_channels(lab, st)
        pg[i] = lab["pg_label"].numpy()
        vm[i] = lab["vm_ref"].numpy()
    return {"var": torch.from_numpy(var), "pg": torch.from_numpy(pg),
            "vm": torch.from_numpy(vm)}


def bus_inputs(static: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """[B, n, 14] model input: static block broadcast + scenario channels."""
    return torch.cat([static.expand(var.shape[0], -1, -1), var], dim=-1)


# ---------------------------------------------------------------------------
# Evaluation through the terminal power flow
# ---------------------------------------------------------------------------
def score_decisions(decide: Callable[[int], tuple], hold_l: List[Dict],
                    st: Dict) -> Dict:
    """Score bus-level decisions on the hold-out through the PV power flow.

    `decide(i)` returns (Pg per bus, Vset per bus) for hold-out scenario i as
    float64 CPU tensors. The channels are those of the PowerForm rows.
    """
    ch = {k: [] for k in ("v_mape", "th_mae", "pg_mae", "qg_mae", "cost_mape")}
    n_nonconv = 0
    gen = st["gen_df"]
    ns_df = gen[~gen.is_slack_gen.astype(bool)]
    sl_df = gen[gen.is_slack_gen.astype(bool)]
    sl = st["slack"]
    sl_i = int(np.where(sl.numpy())[0][0])
    t0 = time.time()
    with torch.no_grad():
        for i, lab in enumerate(hold_l):
            pgb, vset = decide(i)
            grid = dataclasses.replace(st["grid"], p_load=lab["p_load"],
                                       q_load=lab["q_load"])
            vm_set = torch.where(st["m_free"], vset.clamp(*VSET_CLAMP),
                                 torch.ones_like(vset))
            pv = st["m_free"] & ~sl
            pf = transmission_pf_newton_pv(grid, pgb, vm_set, pv,
                                           n_iter=PF_ITERS)
            if float(pf["residual"]) > PF_TOL:
                n_nonconv += 1

            V = pf["V"]
            vm, va = V.abs(), torch.angle(V)
            mn = st["m_node"]
            ch["v_mape"].append(float(((vm - lab["vm_ref"]).abs()
                                       / lab["vm_ref"]).mean()) * 100)
            ch["th_mae"].append(float(np.rad2deg(
                float((va - lab["va_ref"]).abs().mean()))))
            ch["pg_mae"].append(float((pgb[mn] - lab["pg_label"][mn]).abs().mean()))
            ch["qg_mae"].append(float((pf["q_g"][mn] - lab["qg_label"][mn]).abs().mean()))
            c, c_sl = predicted_cost(pgb.numpy() * BASE_MVA,
                                     float(pf["p_slack"][sl_i]) * BASE_MVA,
                                     st, ns_df, sl_df)
            ref = max(lab["cost_ref"], 1e-9)
            ch["cost_mape"].append(abs(c + c_sl - ref) / ref * 100)
            report_progress(i + 1, len(hold_l), t0)

    res = {k: float(np.mean(v)) for k, v in ch.items()}
    res.update({"n_nonconverged": n_nonconv,
                "per_scenario_cost_mape": ch["cost_mape"]})
    return res


def bus_backbone_decider(model: nn.Module, static: torch.Tensor,
                         hold_var: torch.Tensor, device) -> Callable:
    def decide(i: int):
        x = bus_inputs(static, hold_var[i:i + 1].to(device))
        p_out, v_out = model(x)
        return p_out[0].double().cpu(), v_out[0].double().cpu()
    return decide


def gridsfm_decider(model: nn.Module, hold_paths: List[Path], st: Dict,
                    device) -> Callable:
    from powerform.model.baselines.gridsfm import GraphPrep, decision_outputs
    prep = GraphPrep()

    def decide(i: int):
        pgb, vset = decision_outputs(model, [prep(hold_paths[i])], st, device)
        return pgb[0].double().cpu(), vset[0].double().cpu()
    return decide


# ---------------------------------------------------------------------------
# Training: decision recipe
# ---------------------------------------------------------------------------
def train_decision(args) -> None:
    """MLP, GNN or GridSFM under the decision recipe."""
    device = resolve_device(args.device)
    seed = args.seed
    raw = raw_dir_of(args.data_dir)
    st = build_static(raw)
    m_node = st["m_node"].to(device)
    m_free = st["m_free"].to(device)
    train_paths, hold_paths = pyg_split(args.pyg_dir)
    train_paths = spread(train_paths, args.n_train)
    print(f"device={device}  train={len(train_paths)}  holdout="
          f"{len(hold_paths)}  budget={args.steps} steps x batch {BATCH}",
          flush=True)

    hold_l = [scenario_labels(p, st) for p in hold_paths]
    config = decision_config(args.arch)
    if args.arch == "gridsfm":
        from powerform.model.baselines.gridsfm import GraphPrep, decision_outputs
        prep = GraphPrep()
        train_l = [scenario_labels(p, st) for p in train_paths]
        torch.manual_seed(seed)
        model = build_gridsfm_decision().to(device)
    else:
        ctx = build_grid_context(st, raw, device)
        tr = bus_feature_corpus(train_paths, st)
        torch.manual_seed(seed)
        model = build_bus_backbone(config, ctx).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_par:,} parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=ETA_MIN)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        sel = rng.choice(len(train_paths), size=BATCH, replace=False)
        if args.arch == "gridsfm":
            pgb, vset = decision_outputs(
                model, [prep(train_paths[i]) for i in sel], st, device)
            pg_l = torch.stack([train_l[i]["pg_label"] for i in sel]
                               ).to(device).to(pgb.dtype)
            vm_l = torch.stack([train_l[i]["vm_ref"] for i in sel]
                               ).to(device).to(vset.dtype)
        else:
            sel_t = torch.from_numpy(sel)
            pgb, vset = model(bus_inputs(ctx["static"],
                                         tr["var"][sel_t].to(device)))
            pg_l = tr["pg"][sel_t].to(device)
            vm_l = tr["vm"][sel_t].to(device)
        loss = decision_loss(pgb, vset, pg_l, vm_l, m_node, m_free, BATCH)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if (step + 1) % max(1, args.steps // 20) == 0:
            print(f"  step {step+1}/{args.steps}  loss={float(loss):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    print("evaluating on the hold-out through the terminal power flow ...",
          flush=True)
    model.eval()
    if args.arch == "gridsfm":
        decide = gridsfm_decider(model, hold_paths, st, device)
    else:
        hold_var = torch.from_numpy(np.stack(
            [scenario_channels(lab, st) for lab in hold_l]))
        decide = bus_backbone_decider(model, ctx["static"], hold_var, device)
    res = score_decisions(decide, hold_l, st)
    save_checkpoint(args.ckpt_path, {"model": model.state_dict(), "config": config})
    save_record(args.record_path, {
        "seed": seed, "n_train": len(train_paths),
        "samples": args.steps * BATCH, "params": n_par, "train_s": train_time,
        **res})


# ---------------------------------------------------------------------------
# Training: state regression
# ---------------------------------------------------------------------------
def state_row_setup(args, device) -> Dict:
    """Everything the state row's fine-tuning starts from, in the order that fixes
    the RNG stream: seed, split, validation loader, reseed, training set and
    loader, the randomly initialized stock backbone, and its validation pass.
    Iterating a loader draws from torch's generator, so that pass is part of the
    sequence the first training batch depends on."""
    from torch_geometric.loader import DataLoader

    from powerform.model.baselines import gridsfm as G
    from powerform.third_party.gridsfm.eval import eval_pass
    from powerform.third_party.gridsfm.synthetic import SyntheticMixedDataset

    seed = G.STATE_SEED_OF.get(args.seed, args.seed)
    torch.manual_seed(seed)
    paths, _ = pyg_split(args.pyg_dir)
    train_paths, val_paths = G.state_split(paths)
    train_paths = train_paths[:args.n_train] if args.n_train else train_paths
    transform = G.GridTransform()
    val_loader = DataLoader(G.PygJsonOPFDataset(val_paths, transform=transform),
                            batch_size=G.STATE_BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
    base_train = G.PygJsonOPFDataset(train_paths)
    torch.manual_seed(seed)
    train_ds = SyntheticMixedDataset(base_train, seed=seed, transform=transform,
                                     mode_weights=G.STATE_MODE_WEIGHTS,
                                     infeas_prob=G.STATE_INFEAS_PROB)
    train_loader = DataLoader(train_ds, batch_size=G.STATE_BATCH_SIZE,
                              shuffle=True, num_workers=args.num_workers,
                              persistent_workers=False)
    with default_dtype(torch.float32):
        model = G.GridTransformerBackbone().to(device)
    model.train()
    eval_pass(model, val_loader, device=device)
    return {"train_paths": train_paths, "val_paths": val_paths,
            "val_loader": val_loader, "train_loader": train_loader,
            "model": model}


def train_state_gridsfm(args) -> None:
    """GridSFM's fine-tuning loop from random init, 3 epochs."""
    from powerform.model.baselines import gridsfm as G
    from powerform.third_party.gridsfm.checkpoint import _hash_state_dict
    from powerform.third_party.gridsfm.finetune_opfdata import finetune_opfdata

    device = resolve_device(args.device)
    s = state_row_setup(args, device)
    model = s["model"]
    train_paths, val_paths = s["train_paths"], s["val_paths"]
    train_loader, val_loader = s["train_loader"], s["val_loader"]
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device={device}  train={len(train_paths)}  val={len(val_paths)}  "
          f"batch_size={G.STATE_BATCH_SIZE}  epochs={G.STATE_EPOCHS}  "
          f"{n_par:,} parameters", flush=True)
    t0 = time.time()
    finetune_opfdata(
        model, train_loader, val_loader=val_loader, epochs=G.STATE_EPOCHS,
        lr=G.STATE_LR, weight_decay=G.STATE_WEIGHT_DECAY,
        on_epoch_end=lambda e: print(
            f"  ep {e['epoch']:02d}  train_loss={e['train_loss']:.4f}  "
            f"val_loss={e['val_loss']:.4f}  t={e['elapsed_s']:.0f}s",
            flush=True))
    train_time = time.time() - t0
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # `state_dict` and `metadata.hash` are what GridSFM's own loader reads.
    save_checkpoint(args.ckpt_path, {
        "state_dict": state, "metadata": {"hash": _hash_state_dict(state)},
        "config": {"arch": "gridsfm", "recipe": "state"}})
    res = score_state_gridsfm(args.ckpt_path, args, device)
    save_record(args.record_path, {
        "seed": args.seed, "n_train": len(train_paths),
        "samples": G.STATE_EPOCHS * len(train_paths), "params": n_par,
        "train_s": train_time, **res})


def score_state_gridsfm(ckpt: Path, args, device,
                        limit: Optional[int] = None) -> Dict:
    """A state-row checkpoint scored on the hold-out state it emits.

    The forward always runs whole batches of `STATE_EVAL_BATCH`, because its output
    moves with the composition of the batch at the rounding level of float32. The
    first `limit` scores are therefore the first `limit` of the full run.
    """
    from powerform.model.baselines import gridsfm as G
    _, hold_paths = pyg_split(args.pyg_dir)
    n = limit or len(hold_paths)
    model = G.load_state_row(ckpt, device)
    rows = []
    t0 = time.time()
    for i in range(0, n, G.STATE_EVAL_BATCH):
        chunk = hold_paths[i:i + G.STATE_EVAL_BATCH]
        for pred, path in zip(G.emitted_states(model, chunk, device), chunk):
            rows.append(G.score_claimed(pred, G.scenario_targets(path)))
        report_progress(min(len(rows), n), n, t0, every=4 * G.STATE_EVAL_BATCH)
    rows = rows[:n]
    res = {k: float(np.mean([r[k] for r in rows]))
           for k in ("v_mape", "th_mae", "pg_mae", "qg_mae", "cost_mape")}
    res["per_scenario_cost_mape"] = [r["cost_mape"] for r in rows]
    return res


# ---------------------------------------------------------------------------
# Entry points used by training/case500.py
# ---------------------------------------------------------------------------
def load_decision_baseline(blob: Dict, st: Dict, raw, device):
    """A trained MLP, GNN or GridSFM decision checkpoint, rebuilt from its config."""
    if blob["config"]["arch"] == "gridsfm":
        model, ctx = build_gridsfm_decision(), None
    else:
        ctx = build_grid_context(st, raw, device)
        model = build_bus_backbone(blob["config"], ctx)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, ctx


def rescore_baseline(arch: str, recipe: str, ckpt: Path, args, device,
                     limit: Optional[int] = None) -> Dict:
    """Re-score one baseline checkpoint on the Table 1 hold-out, in float32."""
    with default_dtype(torch.float32):
        if arch == "graphkit":
            from powerform.model.baselines.graphkit import rescore_graphkit
            return rescore_graphkit(recipe, ckpt, args, device, limit=limit)
        if recipe == "state":
            return score_state_gridsfm(ckpt, args, device, limit=limit)
        raw = raw_dir_of(args.data_dir)
        st = build_static(raw)
        _, hold_paths = pyg_split(args.pyg_dir)
        hold_paths = hold_paths[:limit] if limit else hold_paths
        hold_l = [scenario_labels(p, st) for p in hold_paths]
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        model, ctx = load_decision_baseline(blob, st, raw, device)
        if arch == "gridsfm":
            decide = gridsfm_decider(model, hold_paths, st, device)
        else:
            hold_var = torch.from_numpy(np.stack(
                [scenario_channels(lab, st) for lab in hold_l]))
            decide = bus_backbone_decider(model, ctx["static"], hold_var,
                                          device)
        return score_decisions(decide, hold_l, st)


def train_baseline(args) -> None:
    """Train one baseline row. The row runs with float32 as the process default,
    so every tensor created without an explicit dtype is float32."""
    with default_dtype(torch.float32):
        if args.arch == "graphkit":
            from powerform.model.baselines.graphkit import train_graphkit
            train_graphkit(args)
        elif args.recipe == "state":
            train_state_gridsfm(args)
        else:
            train_decision(args)
