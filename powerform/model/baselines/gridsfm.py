"""GridSFM rows of Table 1: GridSFM's backbone from random init, under two recipes.

GridSFM (Microsoft, MIT, vendored under `third_party/gridsfm/`) is used as an
architecture only; its released weights are not. Both rows build
`GridTransformerBackbone` with its default hyperparameters and read each scenario
from its `.pyg.json` export through GridSFM's own loader and transforms (cycle
basis, Hodge PE), so the inputs are what GridSFM sees.

Decision row (`DecisionHeadGridSFM`): the trunk, the Hodge PE, the graph context and
every block are unchanged; only the outputs and the loss differ.

  * `head_Pg` emits per-generator power, `leaky_clip`ped into [Pmin, Pmax] and summed
    to bus over non-slack units (`bus_pg`), so the bus total lands in PowerForm's bus
    box by construction.
  * `head_V` emits a per-bus value `leaky_clip`ped into the vm box, read as a voltage
    setpoint.
  * `head_theta` with its DC prior, `head_Qg` and `head_feas` are deleted: theta and
    Qg follow from the power flow here, and the reported parameter count (15.10M of
    the stock 15.15M) is then the count actually trained.

The head is single-shot with no unrolled trajectory, so only its final output is
supervised (t_k = 1).

State-regression row: the stock five-head backbone trained with GridSFM's published
fine-tuning loop (`finetune_opfdata`: AdamW 1e-4, weight decay 1e-4, batch 8,
`SyntheticMixedDataset` with mode weights (0.25, 0.20, 0.20, 0.35) and infeasibility
probability 0.3) for 3 epochs of the 29,237-scenario train part, about 88k samples.
The last 250 training scenarios are its validation probe. The row is scored on the
state it emits, with cost = sum over all in-service units of c2 Pg^2 + c1 Pg + c0
against the label objective.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from powerform.data.datakit_loader import HOLDOUT_START
from powerform.third_party.gridsfm.blocks import _leaky_clip, _nan_to_zero
from powerform.third_party.gridsfm.cycle_basis import (
    CycleBasisCache, prepare_for_grid_transformer_,
)
from powerform.third_party.gridsfm.data import (
    _fill_missing_node_tensors, batch_data_list, load_pyg_json,
    prepare_for_inference,
)
from powerform.third_party.gridsfm.model import (
    _G_CTX_DIM, GridTransformerBackbone,
)
from powerform.third_party.gridsfm.pe_features import (
    LaplacianFactorizationCache, attach_pe_features_,
)
from powerform.third_party.gridsfm.schema import (
    BUS_VMAX_IDX, BUS_VMIN_IDX, GEN_CP0_IDX, GEN_CP1_IDX, GEN_CP2_IDX,
    GEN_PMAX_IDX, GEN_PMIN_IDX,
)

RAD2DEG = 180.0 / np.pi

# GridSFM's fine-tuning recipe (state-regression row).
STATE_BATCH_SIZE = 8
STATE_MODE_WEIGHTS = (0.25, 0.20, 0.20, 0.35)
STATE_INFEAS_PROB = 0.3
STATE_LR = 1e-4
STATE_WEIGHT_DECAY = 1e-4
STATE_EPOCHS = 3
STATE_N_VAL = 250
# Seed 0 of this row runs as GridSFM's own default seed, 42.
STATE_SEED_OF = {0: 42, 1: 1, 2: 2}
# Hold-out inference batch of the state row.
STATE_EVAL_BATCH = 25


# ---------------------------------------------------------------------------
# Decision row
# ---------------------------------------------------------------------------
class DecisionHeadGridSFM(GridTransformerBackbone):
    """GridSFM backbone emitting only the decision variables (Pg, Vset).

    `forward` is inherited; it ends in `_predict_and_write`, the one method
    overridden here.
    """

    def __init__(self):
        super().__init__()
        del self.head_theta, self.head_Qg, self.head_feas

    def _predict_and_write(self, data, h_grid_bus, h_grid_gen,
                           h_grid_global, n_graphs) -> None:
        xg = data["generator"].x
        Pmin = torch.nan_to_num(xg[:, GEN_PMIN_IDX], nan=-1e6)
        Pmax = torch.nan_to_num(xg[:, GEN_PMAX_IDX], nan=+1e6)
        Plo = torch.minimum(Pmin, Pmax)
        Phi = torch.maximum(Pmin, Pmax)

        bus_batch = data["bus"].batch
        g_ctx = getattr(data, "g_ctx", None)
        if g_ctx is None:
            g_ctx = torch.zeros(n_graphs, _G_CTX_DIM,
                                dtype=h_grid_global.dtype,
                                device=h_grid_global.device)
        g_ctx = g_ctx.to(dtype=h_grid_global.dtype, device=h_grid_global.device)

        h_bus_in = torch.cat(
            [h_grid_bus, h_grid_global[bus_batch], g_ctx[bus_batch]], dim=-1)
        gen_batch = data["generator"].batch
        h_gen_in = torch.cat(
            [h_grid_gen, h_grid_global[gen_batch], g_ctx[gen_batch]], dim=-1)

        # Pg per generator, in GridSFM's box and leaky clip
        Pg_raw = _nan_to_zero(self.head_Pg(h_gen_in).squeeze(-1))
        Pg_mid = _nan_to_zero(0.5 * (Plo + Phi))
        Pg = _nan_to_zero(
            _leaky_clip(Pg_raw + Pg_mid, Plo, Phi, leak=self.leaky_alpha))

        # Vset per bus, in the vm box and GridSFM's leaky clip
        v_setpoint = getattr(data["bus"], "v_setpoint", None)
        if v_setpoint is None:
            v_setpoint = torch.zeros(h_bus_in.size(0), 1,
                                     dtype=h_bus_in.dtype,
                                     device=h_bus_in.device)
        v_setpoint = v_setpoint.to(dtype=h_bus_in.dtype, device=h_bus_in.device)
        V_raw = _nan_to_zero(
            self.head_V(torch.cat([h_bus_in, v_setpoint], dim=-1)).squeeze(-1))
        bus_x = data["bus"].x
        Vmin = torch.nan_to_num(bus_x[:, BUS_VMIN_IDX], nan=0.95)
        Vmax = torch.nan_to_num(bus_x[:, BUS_VMAX_IDX], nan=1.05)
        Vset = _nan_to_zero(
            _leaky_clip(V_raw, torch.minimum(Vmin, Vmax),
                        torch.maximum(Vmin, Vmax), leak=self.leaky_alpha),
            sub_value=1.0)

        data["generator"].pg_pred = Pg
        data["bus"].vset_pred = Vset


def bus_pg(pg_gen: torch.Tensor, st: Dict, n_graphs: int) -> torch.Tensor:
    """Sum per-generator Pg onto buses over non-slack units. [n_graphs, n_bus]."""
    n = st["n"]
    idx = torch.as_tensor(st["gen_bus"][st["gen_nonslack"]], dtype=torch.long,
                          device=pg_gen.device)
    G = int(len(st["gen_bus"]))
    pg = pg_gen.view(n_graphs, G)[:, torch.as_tensor(st["gen_nonslack"],
                                                     device=pg_gen.device)]
    out = torch.zeros(n_graphs, n, dtype=pg.dtype, device=pg.device)
    out.index_add_(1, idx, pg)
    return out


class GraphPrep:
    """`.pyg.json` -> prepared HeteroData, with GridSFM's transform caches.

    Graphs are prepared on demand: one prepared HeteroData is 0.37 MB, so holding
    all 29,487 would need about 10.8 GB. Every scenario shares one topology, so
    the cycle-basis and Laplacian caches make the transform cheap after the first
    call.
    """

    def __init__(self):
        self.cb = CycleBasisCache()
        self.pe = LaplacianFactorizationCache()

    def __call__(self, path: Path):
        d = load_pyg_json(path)
        prepare_for_grid_transformer_(d, cache=self.cb)
        attach_pe_features_(d, cache=self.pe)
        _fill_missing_node_tensors(d)
        return d


def decision_outputs(model: DecisionHeadGridSFM, graphs: list, st: Dict,
                     device) -> tuple:
    """(Pg per bus, Vset per bus), each [n_graphs, n_bus], for prepared graphs."""
    b = batch_data_list(graphs).to(device)
    out = model(b)
    pgb = bus_pg(out["generator"].pg_pred, st, len(graphs))
    vset = out["bus"].vset_pred.view(len(graphs), st["n"])
    return pgb, vset


# ---------------------------------------------------------------------------
# State-regression row: the dataset adapter and the hold-out scoring
# ---------------------------------------------------------------------------
class PygJsonOPFDataset(Dataset):
    """The `.pyg.json` scenarios in the OPFData HeteroData schema.

    Mirrors `torch_geometric.datasets.OPFDataset._process` field for field, so
    GridSFM's code reads it as OPFData. Labels come from the `solution` block;
    `feasible` is 1 because every exported scenario is a converged AC-OPF solve,
    the assumption GridSFM's OPFData adapter makes about OPFData.
    """

    case_name = "case500_goc"

    def __init__(self, paths: List[Path], transform=None):
        if not paths:
            raise ValueError("PygJsonOPFDataset got an empty path list")
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from torch_geometric.data import HeteroData
        obj = json.loads(self.paths[int(idx)].read_text())
        grid, sol, md = obj["grid"], obj["solution"], obj["metadata"]

        d = HeteroData()
        d.x = torch.tensor(grid["context"], dtype=torch.float32).view(-1)
        d.objective = torch.tensor(md["objective"], dtype=torch.float32)

        d["bus"].x = torch.tensor(grid["nodes"]["bus"], dtype=torch.float32)
        d["bus"].y = torch.tensor(sol["nodes"]["bus"], dtype=torch.float32)
        d["generator"].x = torch.tensor(grid["nodes"]["generator"],
                                        dtype=torch.float32)
        d["generator"].y = torch.tensor(sol["nodes"]["generator"],
                                        dtype=torch.float32)
        d["load"].x = torch.tensor(grid["nodes"]["load"], dtype=torch.float32)
        d["shunt"].x = torch.tensor(grid["nodes"]["shunt"], dtype=torch.float32)

        for et in ("ac_line", "transformer"):
            e = grid["edges"][et]
            key = ("bus", et, "bus")
            d[key].edge_index = torch.tensor([e["senders"], e["receivers"]],
                                             dtype=torch.long)
            d[key].edge_attr = torch.tensor(e["features"], dtype=torch.float32)
            d[key].edge_label = torch.tensor(sol["edges"][et]["features"],
                                             dtype=torch.float32)

        for et in ("generator_link", "load_link", "shunt_link"):
            e = grid["edges"][et]
            src = et.split("_")[0]
            senders = torch.tensor(e["senders"], dtype=torch.long)
            receivers = torch.tensor(e["receivers"], dtype=torch.long)
            d[src, et, "bus"].edge_index = torch.stack([senders, receivers], 0)
            d["bus", et, src].edge_index = torch.stack([receivers, senders], 0)

        d.feasible = torch.tensor(1, dtype=torch.long)
        if self.transform is not None:
            d = self.transform(d)
        return d


class GridTransform:
    """Cycle-basis and Hodge PE attachment, with GridSFM's caches.

    A module-level class rather than a closure because Windows DataLoader workers
    use `spawn`, which pickles the dataset and hence the transform. The caches are
    built on first call, so each worker process makes its own.
    """

    def __init__(self):
        self._cb: Optional[CycleBasisCache] = None
        self._pe: Optional[LaplacianFactorizationCache] = None

    def __call__(self, d):
        if self._cb is None:
            self._cb = CycleBasisCache()
            self._pe = LaplacianFactorizationCache()
        prepare_for_grid_transformer_(d, cache=self._cb)
        attach_pe_features_(d, cache=self._pe)
        return d


def state_split(train_paths: List[Path], n_val: int = STATE_N_VAL):
    """GridSFM's train/val split inside the training part: val is the tail."""
    if any(int(p.stem.replace(".pyg", "").split("_")[1]) >= HOLDOUT_START
           for p in train_paths):
        raise SystemExit("training paths contain hold-out scenario ids")
    return train_paths[:-n_val], train_paths[-n_val:]


def scenario_targets(path: Path) -> Dict[str, np.ndarray]:
    """Labels, cost coefficients and bus-aggregation index for one scenario."""
    obj = json.loads(Path(path).read_text())
    gen_x = np.asarray(obj["grid"]["nodes"]["generator"], dtype=float)
    sol_bus = np.asarray(obj["solution"]["nodes"]["bus"], dtype=float)
    sol_gen = np.asarray(obj["solution"]["nodes"]["generator"], dtype=float)
    md = obj["metadata"]

    gen_bus = np.asarray(md["gen_bus_map"], dtype=int) - 1
    is_slack_gen = np.asarray(md["gen_is_slack"], dtype=bool)
    bus_type = np.asarray(obj["grid"]["nodes"]["bus"], dtype=float)[:, 1]
    n_bus = len(bus_type)

    # Buses carrying at least one non-slack unit, excluding the reference bus:
    # the definition of the parquet loader.
    m_node = np.zeros(n_bus, dtype=bool)
    m_node[gen_bus[~is_slack_gen]] = True
    m_node &= bus_type != 3

    def to_bus(v: np.ndarray) -> np.ndarray:
        out = np.zeros(n_bus, dtype=float)
        np.add.at(out, gen_bus[~is_slack_gen], v[~is_slack_gen])
        return out

    return {
        "theta_ref": sol_bus[:, 0], "vm_ref": sol_bus[:, 1],
        "pg_bus_ref": to_bus(sol_gen[:, 0]), "qg_bus_ref": to_bus(sol_gen[:, 1]),
        "c2": gen_x[:, GEN_CP2_IDX], "c1": gen_x[:, GEN_CP1_IDX],
        "c0": gen_x[:, GEN_CP0_IDX],
        "gen_bus": gen_bus, "is_slack_gen": is_slack_gen,
        "m_node": m_node, "bus_type": bus_type,
        "cost_ref": float(md["objective"]),
    }


def score_claimed(pred: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray]
                  ) -> Dict[str, float]:
    """Channels of an emitted state, with no power flow: the state row's scoring."""
    vm, th = pred["V"], pred["theta"]
    m = tgt["m_node"]

    def to_bus(v: np.ndarray) -> np.ndarray:
        out = np.zeros(len(tgt["bus_type"]), dtype=float)
        np.add.at(out, tgt["gen_bus"][~tgt["is_slack_gen"]], v[~tgt["is_slack_gen"]])
        return out

    pg_bus, qg_bus = to_bus(pred["Pg"]), to_bus(pred["Qg"])

    # The JSON cost coefficients are pu-referred (c2_pu = c2_mw * 100^2,
    # c1_pu = c1_mw * 100), so evaluating them at Pg in pu gives currency units.
    cost = float((tgt["c2"] * pred["Pg"] ** 2
                  + tgt["c1"] * pred["Pg"] + tgt["c0"]).sum())

    return {
        "v_mape": float(np.mean(np.abs(vm - tgt["vm_ref"]) / tgt["vm_ref"]) * 100),
        "th_mae": float(np.mean(np.abs(th - tgt["theta_ref"])) * RAD2DEG),
        "pg_mae": float(np.mean(np.abs(pg_bus[m] - tgt["pg_bus_ref"][m]))),
        "qg_mae": float(np.mean(np.abs(qg_bus[m] - tgt["qg_bus_ref"][m]))),
        "cost_mape": float(abs(cost - tgt["cost_ref"])
                           / max(abs(tgt["cost_ref"]), 1e-9) * 100),
    }


def emitted_states(model: GridTransformerBackbone, paths: List[Path],
                   device) -> List[Dict[str, np.ndarray]]:
    """The stock backbone's emitted state for a batch of scenarios."""
    from torch_geometric.utils import unbatch
    prepared = [prepare_for_inference(load_pyg_json(p)) for p in paths]
    batch = batch_data_list(prepared).to(device)
    with torch.no_grad():
        out = model(batch)
    bus_pred, gen_pred = out["bus"].pred, out["generator"].pred
    th = unbatch(bus_pred[:, 0], out["bus"].batch)
    V = unbatch(bus_pred[:, 1], out["bus"].batch)
    Pg = unbatch(gen_pred[:, 0], out["generator"].batch)
    Qg = unbatch(gen_pred[:, 1], out["generator"].batch)
    return [
        {"theta": th[i].cpu().numpy().astype(float),
         "V": V[i].cpu().numpy().astype(float),
         "Pg": Pg[i].cpu().numpy().astype(float),
         "Qg": Qg[i].cpu().numpy().astype(float)}
        for i in range(len(paths))
    ]


def load_state_row(ckpt: Path, device) -> GridTransformerBackbone:
    """A state-row checkpoint, through GridSFM's own hash-checking loader."""
    from powerform.third_party.gridsfm.checkpoint import load_model
    return load_model(str(ckpt), device=device)
