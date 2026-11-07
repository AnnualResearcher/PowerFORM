"""GridSFM with the line-status track of the reconfiguration study (App. D).

GridSFM has no line-status gate, so this module adds one together with the per-line
network that sets it, as close as GridSFM's structure allows to what
`powerform/model/line_track.py` does on PowerFORM:

  * The same `LineStatusTrack`: a zero-initialised update per block and one shared
    readout, so ell = 0, the gate is exactly 1 and the first forward is GridSFM's own.
    The track reads the normalised latents of the two endpoint buses and of the branch,
    plus the log of the live gate (3 x 128 + 1 inputs; GridSFM keeps one latent per bus,
    not separate V and S).
  * The gate acts inside every block's message passing: the branch latents are
    multiplied by exp(ell) after the block's LayerNorm, just before its HeteroConv.
    Before the LayerNorm a gate could not close a line, since the LayerNorm restores the
    scale and a zero latent comes out as its bias. The per-type self-attention, over
    every node type, the positional features and the fusion layer are left as published.
  * GridSFM's forward runs unchanged. Only `block.gnn` is wrapped and
    `_predict_and_write` replaced, since no decision head is used here.

The task binding follows `powerform/data/binding.py` on GridSFM's own input channels:
the fixed dispatch goes on the generators' Pmax (Pmin = 0, slack units 0), and the
voltage box on the buses' Vmin and Vmax. An outaged line is deleted from the graph
before GridSFM's transforms, the way `open_lines` deletes it from the instance, so both
models choose among the same lines in the same order (ac lines, then transformers).

The pretrained arm starts from the GridSFM backbone generalist of Table 2,
`opf_generalist_gridsfm_seed0.pt`.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from powerform.data.opfdata import iter_raw
from powerform.model.baselines.common import default_dtype
from powerform.model.line_track import LineStatusTrack
from powerform.paths import CHECKPOINTS
from powerform.third_party.gridsfm.cycle_basis import (
    CycleBasisCache, prepare_for_grid_transformer_,
)
from powerform.third_party.gridsfm.data import _fill_missing_node_tensors
from powerform.third_party.gridsfm.model import GridTransformerBackbone
from powerform.third_party.gridsfm.pe_features import (
    LaplacianFactorizationCache, attach_pe_features_,
)
from powerform.third_party.gridsfm.schema import (
    BUS_VMAX_IDX, BUS_VMIN_IDX, GEN_PMAX_IDX, GEN_PMIN_IDX,
)

GENERALIST = CHECKPOINTS / "opf_generalist_gridsfm_seed0.pt"
FETCH = "python scripts/download_checkpoints.py --group gridsfm_reconfig"
HEADS = ("head_theta", "head_V", "head_Pg", "head_Qg", "head_feas")
AC, TR = ("bus", "ac_line", "bus"), ("bus", "transformer", "bus")


class BranchTrack(LineStatusTrack):
    N_INPUT_BLOCKS = 3          # bus[u], bus[v], branch


class GateState:
    """What the wrapped blocks share during one forward. Not a module, so the wrappers
    do not register the track a second time."""

    ell: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    n_ac: int
    track: BranchTrack


class GatedConv(nn.Module):
    """`block.gnn` with the gate applied to the normalised branch latents."""

    def __init__(self, inner: nn.Module, k: int, state: GateState) -> None:
        super().__init__()
        self.inner, self.k = inner, k
        self._state = [state]

    def forward(self, x_pre: dict, edge_index_dict: dict, **kwargs) -> dict:
        s = self._state[0]
        gate = torch.exp(s.ell).to(x_pre["bus"].dtype)
        branch = torch.cat([x_pre["branch_ac"], x_pre["branch_tr"]], dim=0)
        e_in = torch.cat([x_pre["bus"][s.u], x_pre["bus"][s.v], branch], dim=-1)
        gated = dict(x_pre)
        gated["branch_ac"] = x_pre["branch_ac"] * gate[:s.n_ac, None]
        gated["branch_tr"] = x_pre["branch_tr"] * gate[s.n_ac:, None]
        out = self.inner(gated, edge_index_dict, **kwargs)
        s.ell = s.ell + s.track(self.k, {"e_in": e_in, "z": gate})
        return out


class GatedGridSFM(GridTransformerBackbone):
    """The backbone without its heads. `line_logits` runs GridSFM's forward."""

    def __init__(self) -> None:
        super().__init__()
        for head in HEADS:
            delattr(self, head)
        self.z_stream = None
        self.state = GateState()
        self.h_bus = None

    def _predict_and_write(self, data, h_grid_bus, h_grid_gen, h_grid_global, n_graphs) -> None:
        self.h_bus = h_grid_bus

    def attach_track(self, z_d: int) -> None:
        self.z_stream = BranchTrack(self.hidden_dim, self.num_blocks, z_d)
        self.state.track = self.z_stream
        for k, block in enumerate(self.blocks):
            block.gnn = GatedConv(block.gnn, k, self.state)

    def line_logits(self, data, ell0: torch.Tensor | None = None) -> torch.Tensor:
        ac, tr = data[AC].edge_index, data[TR].edge_index
        s = self.state
        s.u, s.v = torch.cat([ac[0], tr[0]]), torch.cat([ac[1], tr[1]])
        s.n_ac = ac.size(1)
        s.ell = (torch.zeros(s.u.size(0), dtype=data["bus"].x.dtype, device=s.u.device)
                 if ell0 is None else ell0)
        self.z_stream.reset()
        self(data)
        return s.ell


def load_generalist(net: GatedGridSFM, path: Path = GENERALIST) -> None:
    """The Table 2 generalist's backbone weights; its heads are not used."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found; fetch it with: {FETCH}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = {k[len("net."):]: v for k, v in blob["model"].items()
             if k.startswith("net.") and not k[len("net."):].startswith(HEADS)}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{Path(path).name} does not fit: missing {missing[:5]}, "
                         f"unexpected {unexpected[:5]}")


def to_hetero(obj: dict):
    """An OPFData scenario as GridSFM's HeteroData, in float32."""
    from torch_geometric.data import HeteroData

    f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    grid, sol, md = obj["grid"], obj["solution"], obj["metadata"]
    d = HeteroData()
    d.x = f32(grid["context"]).view(-1)
    d.objective = f32(md["objective"])
    d["bus"].x, d["bus"].y = f32(grid["nodes"]["bus"]), f32(sol["nodes"]["bus"])
    d["generator"].x, d["generator"].y = f32(grid["nodes"]["generator"]), f32(sol["nodes"]["generator"])
    d["load"].x, d["shunt"].x = f32(grid["nodes"]["load"]), f32(grid["nodes"]["shunt"])
    for et in ("ac_line", "transformer"):
        e = grid["edges"][et]
        key = ("bus", et, "bus")
        d[key].edge_index = torch.tensor([e["senders"], e["receivers"]], dtype=torch.long)
        d[key].edge_attr = f32(e["features"])
        d[key].edge_label = f32(sol["edges"][et]["features"])
    for et in ("generator_link", "load_link", "shunt_link"):
        e = grid["edges"][et]
        src = et.split("_")[0]
        s, r = torch.tensor(e["senders"], dtype=torch.long), torch.tensor(e["receivers"], dtype=torch.long)
        d[src, et, "bus"].edge_index = torch.stack([s, r], 0)
        d["bus", et, src].edge_index = torch.stack([r, s], 0)
    d.feasible = torch.tensor(1, dtype=torch.long)
    return d


def delete_line(d, line: int) -> None:
    """Remove branch `line` of the instance's table: ac lines first, then transformers."""
    n_ac = d[AC].edge_index.size(1)
    key, idx = (AC, line) if line < n_ac else (TR, line - n_ac)
    keep = torch.ones(d[key].edge_index.size(1), dtype=torch.bool)
    keep[idx] = False
    d[key].edge_index = d[key].edge_index[:, keep]
    d[key].edge_attr = d[key].edge_attr[keep]
    d[key].edge_label = d[key].edge_label[keep]


def attach_graphs(pool, grid: str, split: str, outages: list[int]) -> None:
    """Give every instance of `pool` its graph, in the order the pool was read."""
    wanted = set(pool.example_indices)
    graphs = {idx: to_hetero(obj) for idx, obj in iter_raw(grid, "fulltop", split, wanted=wanted)}
    for i, idx in enumerate(pool.example_indices):
        het = graphs[idx]
        if outages[i] >= 0:
            delete_line(het, outages[i])
        pool.insts[i].gs = {"het": het, "prepared": None}


def bind(d, inst, spec) -> None:
    """Put the task's fixed dispatch and voltage box on GridSFM's input channels."""
    slack = inst.tgrid.is_slack.cpu()
    gen_bus = d["generator", "generator_link", "bus"].edge_index[1]
    pg = d["generator"].y[:, 0].clamp_min(0.0)
    pmax = torch.where(slack[gen_bus], torch.zeros_like(pg), pg)
    d["generator"].x[:, GEN_PMIN_IDX] = 0.0
    d["generator"].x[:, GEN_PMAX_IDX] = pmax
    d["bus"].x[:, BUS_VMIN_IDX] = spec.vm_min.detach().cpu().to(torch.float32)
    d["bus"].x[:, BUS_VMAX_IDX] = spec.vm_max.detach().cpu().to(torch.float32)


class GridSFMReconfigNet(nn.Module):
    """`forward(inst, spec)` returns one log-gate per line, like `ReconfigurationNet`."""

    def __init__(self, trunk: GatedGridSFM) -> None:
        super().__init__()
        self.trunk = trunk
        self._cb, self._pe = CycleBasisCache(), LaplacianFactorizationCache()

    def graph(self, inst, spec):
        """The prepared graph of a scenario, built once: the binding and the
        transforms depend only on the scenario."""
        gs = inst.gs
        if gs["prepared"] is None:
            with default_dtype(torch.float32):
                d = gs["het"].clone()
                bind(d, inst, spec)
                prepare_for_grid_transformer_(d, cache=self._cb)
                attach_pe_features_(d, cache=self._pe)
                _fill_missing_node_tensors(d)
            gs["prepared"] = d.to(inst.p_load.device)
        return gs["prepared"]

    def forward(self, inst, spec) -> torch.Tensor:
        return self.trunk.line_logits(self.graph(inst, spec))


def build(arm: str, z_d: int, init: Path = GENERALIST) -> GridSFMReconfigNet:
    """Both arms build the trunk before the track, as `build_model_for` does."""
    trunk = GatedGridSFM()
    if arm == "pretrained":
        load_generalist(trunk, init)
    trunk.attach_track(z_d)
    return GridSFMReconfigNet(trunk).float()
