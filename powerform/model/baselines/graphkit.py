"""graphkit rows of Table 1: gridfm-graphkit's backbone under two recipes.

gridfm-graphkit (GridFM / IBM, Apache-2.0) is installed as its own package at a
pinned commit (`GRAPHKIT_INSTALL`) and runs in its own environment, because it needs
`torch_scatter`, which pins torch. This module turns its model and dataset into the
two rows; every graphkit import happens at call time, so the module imports without
graphkit.

Decision row (`train_graphkit`): graphkit's `GNS_heterogeneous` as its published
case500 config builds it (12 layers, hidden 48, 8 heads, 20,069,187 parameters), from
random init and unmodified. Its forward emits the two decisions directly: per-generator
Pg squashed into [Pmin, Pmax] and per-bus Vm squashed into the vm box, read as a voltage
setpoint. Va stays live inside graphkit's per-layer physics feedback and is not scored;
Qg is decoded analytically. Inputs come from graphkit's own dataset, transforms, masking
and normalizer. Training covers the full 29,487-scenario part (graphkit's train | val
split), 20,000 steps of batch 4, scored through the shared terminal power flow
(`recipe.score_decisions`). Only the final layer's decode is returned, so only it is
supervised (t_k = 1).

State-regression row: graphkit's own Lightning trainer and CLI with its published case500
config, adapted as `graphkit_configs/case500_state_80k.yaml` documents, for 3 epochs of
26,538 scenarios = 79,614 samples. graphkit keeps a 2,949-scenario validation split for
its best-epoch selection. `launch_graphkit_cli` is the launcher, and
`score_graphkit_predictions` scores graphkit's `predict` dumps on the emitted state.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from powerform.data.corpus.datakit_export import require_raw
from powerform.data.datakit_loader import (
    BASE_MVA, HOLDOUT_END, HOLDOUT_START, build_static, raw_dir_of,
    scenario_labels,
)
from powerform.device import resolve_device
from powerform.model.baselines.common import default_dtype, spread
from powerform.model.baselines.recipe import (
    BATCH, ETA_MIN, GRAD_CLIP, LR, VSET_WEIGHT, W_BAL, save_checkpoint, save_record,
    score_decisions,
)

CONFIG_DIR = Path(__file__).resolve().parent / "graphkit_configs"
DECISION_CONFIG = CONFIG_DIR / "case500_decision.yaml"
STATE_CONFIG = CONFIG_DIR / "case500_state_80k.yaml"
NETWORK = "case500_goc"
VAL_RATIO = 0.1

GRAPHKIT_INSTALL = """\
gridfm-graphkit is installed separately (Apache License 2.0), in an environment of its
own, with the versions of powerform/model/baselines/graphkit_requirements.txt. For a GPU
(to re-score only, use `cpu` for `cu126`), from this repository's root:

    pip install -r powerform/model/baselines/graphkit_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 -f https://data.pyg.org/whl/torch-2.9.1+cu126.html
    git clone https://github.com/gridfm/gridfm-graphkit.git
    cd gridfm-graphkit
    git checkout b3d663b62179222c1ebec00ee29f67ea50e68c0b
    pip install --no-deps -e .
"""


def _require_graphkit():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import gridfm_graphkit  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"gridfm-graphkit is not installed ({e}).\n\n"
                         + GRAPHKIT_INSTALL) from e


# ---------------------------------------------------------------------------
# Split files pinned to the Table 1 hold-out
# ---------------------------------------------------------------------------
def make_splits(raw_dir, out_dir) -> Dict[str, int]:
    """graphkit train/val/test index files pinned to the Table 1 hold-out.

    A graphkit dataset index is the parquet `scenario` column. test = ids
    29,500-29,999; val = the last 10 % of the rest by load_scenario_idx (graphkit's
    val_ratio); train = everything else.
    """
    require_raw(Path(raw_dir))
    bus = pd.read_parquet(Path(raw_dir) / "bus_data.parquet",
                          columns=["scenario", "load_scenario_idx"])
    m = bus.groupby("scenario", sort=True)["load_scenario_idx"].first()
    scenario_idx = m.index.to_numpy()
    load_idx = m.to_numpy()
    if scenario_idx.min() != 0 or scenario_idx.max() != len(scenario_idx) - 1:
        raise SystemExit("graphkit requires contiguous scenario ids 0..N-1")
    is_test = load_idx >= HOLDOUT_START
    test = scenario_idx[is_test]
    rest = scenario_idx[~is_test]
    # Ordered by load_scenario_idx, so val is the tail of the training pool, as the
    # hold-out is the tail of the corpus.
    rest_sorted = rest[load_idx[~is_test].argsort()]
    n_val = int(round(len(rest_sorted) * VAL_RATIO))
    val, train = rest_sorted[-n_val:], rest_sorted[:-n_val]
    if len(test) != 500:
        raise SystemExit(f"expected 500 hold-out scenarios, got {len(test)}")
    if (set(train.tolist()) | set(val.tolist())) & set(test.tolist()):
        raise SystemExit("train/val overlaps the hold-out")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in (("train", train), ("val", val), ("test", test)):
        torch.save(torch.tensor(arr, dtype=torch.long), out / f"{name}.pt")
    return {"train": len(train), "val": len(val), "test": len(test)}


def config_with_splits(config: Path, splits_dir: Path, out: Path) -> Path:
    """A copy of `config` whose split path points at `splits_dir`."""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    cfg["data"]["split_from_existing_files"] = str(Path(splits_dir).resolve())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(out)


# ---------------------------------------------------------------------------
# graphkit batch -> the two decisions
# ---------------------------------------------------------------------------
def batch_masks(b) -> Dict[str, torch.Tensor]:
    """Per-bus m_node / m_free and the non-slack gen -> bus scatter index.

    Everything is read off the batch, so this holds for any topology and batch size.
    A generator counts as slack if it sits on a REF bus, which matches the parquet's
    `is_slack_gen` flag exactly on case500_goc (one slack unit, on bus 312, and 112
    non-slack generator buses).
    """
    ref = b.mask_dict["REF"]
    n_bus = b.x_dict["bus"].size(0)
    gen_i, bus_i = b.edge_index_dict[("gen", "connected_to", "bus")]
    gen2bus = torch.zeros(b.x_dict["gen"].size(0), dtype=torch.long,
                          device=gen_i.device)
    gen2bus[gen_i] = bus_i
    ns_gen = ~ref[gen2bus]

    hits = torch.zeros(n_bus, dtype=torch.long, device=ref.device)
    hits.index_add_(0, gen2bus[ns_gen],
                    torch.ones(int(ns_gen.sum()), dtype=torch.long, device=ref.device))
    m_node = hits > 0
    return {"ref": ref, "m_node": m_node, "m_free": m_node | ref,
            "gen2bus": gen2bus, "ns_gen": ns_gen, "n_bus": n_bus}


def bus_pg(pg_gen: torch.Tensor, mk: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum per-generator Pg onto buses over non-slack units. [n_bus_total]."""
    out = torch.zeros(mk["n_bus"], dtype=pg_gen.dtype, device=pg_gen.device)
    out.index_add_(0, mk["gen2bus"][mk["ns_gen"]], pg_gen[mk["ns_gen"]])
    return out


def decisions(out, b, base_mva: float):
    """(Pg per generator in pu, Vset per bus in pu) from graphkit's output.

    The generator head is normalized by the fitted baseMVA (151.17 here, not 100);
    `* base_mva / BASE_MVA` puts it on 100 MVA. `HeteroDataMVANormalizer` leaves Vm
    alone, so it is already pu.
    """
    from gridfm_graphkit.datasets.globals import PG_OUT_GEN, VM_OUT
    pg_pu = out["gen"][:, PG_OUT_GEN] * (base_mva / BASE_MVA)
    vset = out["bus"][:, VM_OUT]
    return pg_pu, vset


def decision_step_loss(model, b, base_mva: float, n_m: int,
                       batch: int) -> torch.Tensor:
    """The decision recipe at t_k = 1 on one graphkit batch.

    graphkit batches several scenarios into one disjoint graph, so the balance term
    is per graph of the batch, summed.
    """
    from gridfm_graphkit.datasets.globals import VM_H
    out = model(b)
    mk = batch_masks(b)
    pg_pu, vset = decisions(out, b, base_mva)
    pgb = bus_pg(pg_pu, mk)
    pg_lab = bus_pg(b.y_dict["gen"][:, 0] * (base_mva / BASE_MVA), mk)
    vm_lab = b.y_dict["bus"][:, VM_H]
    mn, mf = mk["m_node"], mk["m_free"]
    bus_batch = b["bus"].batch
    sup_p = ((pgb[mn] - pg_lab[mn]) ** 2).sum()
    sup_v = ((vset[mf] - vm_lab[mf]) ** 2).sum() * VSET_WEIGHT
    per_graph = torch.zeros(b.num_graphs, dtype=pgb.dtype, device=pgb.device)
    per_graph.index_add_(0, bus_batch[mn], (pgb - pg_lab)[mn])
    bal = (per_graph ** 2).sum()
    return (sup_p + sup_v + W_BAL * bal / n_m) / batch


def build_dataset(cfg_args, data_root: Path, norm_stats: Optional[Path],
                  splits_dir: Path):
    """graphkit's dataset with its transforms and its fitted normalizer."""
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import (
        HeteroGridDatasetDisk)
    from gridfm_graphkit.io.param_handler import (
        get_task_transforms, load_normalizer)
    nz = load_normalizer(cfg_args)
    if norm_stats is not None:
        if not Path(norm_stats).exists():
            raise SystemExit(f"--graphkit-norm-stats {norm_stats} does not "
                             "exist (omit it to refit on the train split)")
        stats = torch.load(norm_stats, map_location="cpu", weights_only=True)
        nz.fit_from_dict(stats[NETWORK])
        src = f"restored from {Path(norm_stats).name}"
    else:
        train_ids = [int(i) for i in torch.load(Path(splits_dir) / "train.pt",
                                                weights_only=True)]
        nz.fit(str(Path(data_root) / "raw"), train_ids)
        src = ("refitted on the train split, which is not the normalizer of the "
               "Table 1 row (baseMVA 151.1737); pass --graphkit-norm-stats to "
               "reproduce the row")
    ds = HeteroGridDatasetDisk(root=str(data_root), data_normalizer=nz,
                               transform=get_task_transforms(cfg_args))
    print(f"dataset={len(ds)}  normalizer baseMVA={nz.baseMVA:.4f} ({src})")
    return ds, nz


def load_splits(ds, splits_dir: Path) -> Dict[str, List[int]]:
    out = {}
    for split in ("train", "val", "test"):
        idx = torch.load(Path(splits_dir) / f"{split}.pt", weights_only=True)
        out[split] = [int(i) for i in idx]
    lsi = ds.load_scenarios[torch.tensor(out["test"])].numpy()
    if len(lsi) != 500 or lsi.min() != HOLDOUT_START or lsi.max() != HOLDOUT_END:
        raise SystemExit(f"test split is not the hold-out: n={len(lsi)} "
                         f"range {lsi.min()}..{lsi.max()}")
    overlap = (set(out["train"]) | set(out["val"])) & set(out["test"])
    if overlap:
        raise SystemExit(f"train/test leakage: {len(overlap)} shared indices")
    return out


def _paths(args) -> Dict[str, Path]:
    data_root = Path(args.data_dir) / NETWORK
    if not args.graphkit_splits:
        raise SystemExit("--graphkit-splits is required (see make_splits)")
    return {"data_root": data_root, "splits": Path(args.graphkit_splits),
            "norm": Path(args.graphkit_norm_stats)
            if args.graphkit_norm_stats else None}


def _cfg(config: Path):
    import yaml
    from gridfm_graphkit.io.param_handler import NestedNamespace
    return NestedNamespace(**yaml.safe_load(Path(config).read_text()))


def graphkit_decider(model, ds, test_idx, nz, device):
    from torch_geometric.data import Batch

    def decide(i: int):
        b = Batch.from_data_list([ds[test_idx[i]]]).to(device)
        mk = batch_masks(b)
        pg_pu, vset = decisions(model(b), b, nz.baseMVA)
        return bus_pg(pg_pu, mk).double().cpu(), vset.double().cpu()
    return decide


# ---------------------------------------------------------------------------
# Decision row: train and re-score
# ---------------------------------------------------------------------------
def graphkit_setup(args) -> Dict:
    """Everything a decision run starts from, in float32 as graphkit runs: seed,
    static grid, dataset and normalizer, splits, then the randomly initialized
    model, which graphkit's `load_model` builds from the config."""
    _require_graphkit()
    from gridfm_graphkit.io.param_handler import load_model
    with default_dtype(torch.float32):
        torch.manual_seed(args.seed)
        p = _paths(args)
        st = build_static(raw_dir_of(args.data_dir))
        cfg_args = _cfg(DECISION_CONFIG)
        ds, nz = build_dataset(cfg_args, p["data_root"], p["norm"],
                               p["splits"])
        splits = load_splits(ds, p["splits"])
        # train | val is the whole non-hold-out corpus: a fixed step budget has no
        # model selection for a validation split to serve.
        train_idx = spread(sorted(set(splits["train"]) | set(splits["val"])),
                           args.n_train)
        model = load_model(cfg_args)
    return {"st": st, "n_m": int(st["m_node"].sum()), "cfg": cfg_args,
            "ds": ds, "nz": nz, "splits": splits, "train_idx": train_idx,
            "test_idx": splits["test"], "model": model}


def train_graphkit(args) -> None:
    from torch_geometric.data import Batch

    device = resolve_device(args.device)
    g = graphkit_setup(args)
    st, n_m, ds, nz = g["st"], g["n_m"], g["ds"], g["nz"]
    train_idx, test_idx = g["train_idx"], g["test_idx"]
    cfg_args = g["cfg"]
    model = g["model"].to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"{cfg_args.model.type}: {n_par:,} parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=ETA_MIN)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        sel = rng.choice(len(train_idx), size=BATCH, replace=False)
        b = Batch.from_data_list([ds[train_idx[i]] for i in sel]).to(device)
        loss = decision_step_loss(model, b, nz.baseMVA, n_m, BATCH)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if (step + 1) % max(1, args.steps // 20) == 0:
            print(f"  step {step+1}/{args.steps}  loss={float(loss):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    train_time = time.time() - t0
    model.eval()
    res = _score(model, ds, nz, test_idx, st, args, device)
    save_checkpoint(args.ckpt_path, {
        "model": model.state_dict(),
        "config": {"arch": "graphkit", "recipe": "decision"}})
    save_record(args.record_path, {
        "seed": args.seed, "n_train": len(train_idx),
        "samples": args.steps * BATCH, "params": n_par, "train_s": train_time,
        **res})


def _score(model, ds, nz, test_idx, st, args, device, limit=None) -> Dict:
    test_idx = test_idx[:limit] if limit else test_idx
    lsi = [int(ds.load_scenarios[i]) for i in test_idx]
    hold_dir = Path(args.pyg_dir) / "holdout_opfdata"
    hold_l = [scenario_labels(hold_dir / f"scenario_{s}.pyg.json", st)
              for s in lsi]
    return score_decisions(graphkit_decider(model, ds, test_idx, nz, device),
                           hold_l, st)


def rescore_graphkit(recipe: str, ckpt: Path, args, device,
                     limit: Optional[int] = None) -> Dict:
    """Re-score a graphkit row: the decision checkpoint, or the state row's `predict`
    output directory passed as `ckpt`."""
    if recipe == "state":
        return score_graphkit_predictions(Path(ckpt), raw_dir_of(args.data_dir),
                                          limit=limit)
    _require_graphkit()
    from gridfm_graphkit.io.param_handler import load_model
    p = _paths(args)
    st = build_static(raw_dir_of(args.data_dir))
    cfg_args = _cfg(DECISION_CONFIG)
    ds, nz = build_dataset(cfg_args, p["data_root"], p["norm"], p["splits"])
    splits = load_splits(ds, p["splits"])
    model = load_model(cfg_args)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    with torch.no_grad():
        return _score(model, ds, nz, splits["test"], st, args, device,
                      limit=limit)


# ---------------------------------------------------------------------------
# State row: graphkit's own CLI, and the score of its prediction dumps
# ---------------------------------------------------------------------------
def launch_graphkit_cli(argv: List[str]) -> None:
    """graphkit's CLI (`train` / `predict`) on a single Windows GPU.

    graphkit forces `DDPStrategy` whenever the accelerator is not mps/cpu and the
    strategy is "auto" or "ddp", and Windows PyTorch has no NCCL. One extra Lightning
    strategy alias, `single_device_cuda`, bound to `SingleDeviceStrategy(device=
    "cuda:0")`, is registered and selected by the config; graphkit passes it through
    to `L.Trainer`. Model, losses, optimizer and schedule are graphkit's.
    """
    _require_graphkit()
    from lightning.pytorch.strategies import (SingleDeviceStrategy,
                                              StrategyRegistry)
    name = "single_device_cuda"
    if name not in StrategyRegistry.available_strategies():
        StrategyRegistry.register(
            name, SingleDeviceStrategy,
            description="SingleDeviceStrategy pinned to cuda:0 (Windows, no NCCL)",
            device="cuda:0")
    from gridfm_graphkit.__main__ import main as graphkit_main
    sys.argv = ["gridfm_graphkit"] + list(argv)
    graphkit_main()


def score_graphkit_predictions(pred_dir: Path, raw_dir,
                               limit: Optional[int] = None) -> Dict:
    """Channels of graphkit's emitted hold-out state, from its `predict` dumps.

    Cost = per-unit quadratic cost over all in-service units, slack included, at the
    emitted Pg, against the same quantity at the label Pg. `limit` scores the first
    `limit` scenarios of the dump.
    """
    raw = Path(raw_dir)
    require_raw(raw)
    bus = pd.read_parquet(Path(pred_dir) / "case500_goc_predictions.parquet")
    gen = pd.read_parquet(Path(pred_dir) / "case500_goc_gen_predictions.parquet")
    raw_bus = pd.read_parquet(raw / "bus_data.parquet",
                              columns=["scenario", "load_scenario_idx"])
    m = raw_bus.groupby("scenario", sort=True)["load_scenario_idx"].first()
    bus["lsi"] = bus["scenario"].map(m)
    gen["lsi"] = gen["scenario"].map(m)
    lsi = np.sort(bus["lsi"].unique())
    if len(lsi) != 500 or lsi.min() != HOLDOUT_START or lsi.max() != HOLDOUT_END:
        raise SystemExit(f"prediction dump is not the hold-out: n={len(lsi)} "
                         f"range {lsi.min()}..{lsi.max()}")
    g0 = pd.read_parquet(raw / "gen_data.parquet",
                         columns=["load_scenario_idx", "idx", "bus",
                                  "in_service", "is_slack_gen"])
    g0 = g0[g0.load_scenario_idx == g0.load_scenario_idx.min()]
    g0 = g0[g0.in_service > 0]
    slack_gen_idx = set(g0.loc[g0.is_slack_gen > 0, "idx"].astype(int))
    nonslack_buses = sorted(set(g0.loc[g0.is_slack_gen == 0, "bus"].astype(int)))
    n_bus = int(bus["bus"].max()) + 1
    ref_row = bus[bus["scenario"] == bus["scenario"].iloc[0]].sort_values("bus")
    is_ref_bus = ref_row["REF"].values.astype(bool)
    m_node = np.zeros(n_bus, dtype=bool)
    m_node[nonslack_buses] = True
    m_node &= ~is_ref_bus
    gen_ns = gen[~gen["idx"].astype(int).isin(slack_gen_idx)]

    rows = []
    for sid, b in bus.groupby("scenario"):
        if limit and len(rows) == limit:
            break
        b = b.sort_values("bus")
        vm_p, vm_r = b["Vm_pred"].values, b["Vm_target"].values
        va_p, va_r = b["Va_pred"].values, b["Va_target"].values
        qg_p, qg_r = b["Qg_pred"].values, b["Qg_target"].values
        gs = gen_ns[gen_ns["scenario"] == sid]
        pg_bus_p = np.zeros(n_bus)
        pg_bus_r = np.zeros(n_bus)
        np.add.at(pg_bus_p, gs["bus"].values.astype(int), gs["p_mw_pred"].values)
        np.add.at(pg_bus_r, gs["bus"].values.astype(int), gs["p_mw_target"].values)
        ga = gen[gen["scenario"] == sid]
        c2, c1, c0 = (ga["cp2_eur_per_mw2"].values, ga["cp1_eur_per_mw"].values,
                      ga["cp0_eur"].values)
        cost_p = float((c2 * ga["p_mw_pred"].values ** 2
                        + c1 * ga["p_mw_pred"].values + c0).sum())
        cost_r = float((c2 * ga["p_mw_target"].values ** 2
                        + c1 * ga["p_mw_target"].values + c0).sum())
        rows.append({
            "v_mape": float(np.mean(np.abs(vm_p - vm_r) / vm_r) * 100),
            "th_mae": float(np.mean(np.abs(va_p - va_r)) * 180.0 / np.pi),
            "pg_mae": float(np.mean(np.abs(pg_bus_p[m_node]
                                           - pg_bus_r[m_node])) / BASE_MVA),
            "qg_mae": float(np.mean(np.abs(qg_p[m_node]
                                           - qg_r[m_node])) / BASE_MVA),
            "cost_mape": float(abs(cost_p - cost_r) / max(abs(cost_r), 1e-9) * 100),
        })
    df = pd.DataFrame(rows)
    res = {c: float(df[c].mean())
           for c in ("v_mape", "th_mae", "pg_mae", "qg_mae", "cost_mape")}
    res["per_scenario_cost_mape"] = [r["cost_mape"] for r in rows]
    return res


if __name__ == "__main__":
    # graphkit's CLI with the single-GPU strategy registered, e.g.
    #   python -m powerform.model.baselines.graphkit train --config CFG \
    #       --data_path D --exp_name graphkit_state --run_name s0
    launch_graphkit_cli(sys.argv[1:])
