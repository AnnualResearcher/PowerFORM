"""What the MLP and GNN rows share apart from the backbone.

Both see the input features PowerForm sees, so no row gains or loses information:
9 static per-bus channels (shunts, vm box, p box, slack flag, decision flag, BFS
depth) and 5 scenario channels (raw and per-instance standardized loads, slack
|V|). The raw loads stay because standardizing alone erases the load scale, which
the dispatch depends on. Both read out through `DecisionHead`, the form of
PowerForm's `DecisionReadout`.

Both run in float32, as a 15M-parameter model in float64 is far slower on a GPU;
everything handed to the power flow is cast to float64 at the call site.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from powerform.data.datakit_loader import BASE_MVA, HOLDOUT_START
from powerform.model.spec import standardize

N_BUS_FEATS = 14        # 9 static + 5 scenario-varying
N_EDGE_FEATS = 9        # 8 line constants + direction sign


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Run a block under `dtype` as the global default, then restore it."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def spread(items: Sequence, n: int) -> list:
    """`n` items spread evenly over `items`; 0 keeps them all. This is what
    `--n-train` selects, except on the GridSFM state row."""
    if not n:
        return list(items)
    return [items[i] for i in
            np.linspace(0, len(items) - 1, n).round().astype(int)]


def report_progress(done: int, total: int, t0: float, every: int = 50) -> None:
    """A line every `every` scenarios: a full evaluation takes minutes on a CPU."""
    if done % every == 0 or done == total:
        print(f"  scored {done}/{total} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Static grid context: per-bus static features and the message-passing graph
# ---------------------------------------------------------------------------
def build_grid_context(st: Dict, raw_dir, device) -> Dict:
    """Scenario-invariant tensors: bus feature block, edge index, edge feats.

    `st` is `datakit_loader.build_static(raw_dir)`. Boxes follow PowerForm's
    TaskSpec: the p box is the per-bus sum of non-slack unit boxes with the slack
    bus zeroed, since the slack injection is a consequence of the power flow and
    never a decision; the second box is the bus vm box.
    """
    n = st["n"]
    grid = st["grid"]
    slack_np = st["slack"].numpy()

    bus = (pd.read_parquet(Path(raw_dir) / "bus_data.parquet",
                           filters=[("load_scenario_idx", "==",
                                     float(HOLDOUT_START))])
           .sort_values("bus").reset_index(drop=True))
    vm_min = bus.min_vm_pu.values.astype(np.float64)
    vm_max = bus.max_vm_pu.values.astype(np.float64)

    gen = st["gen_df"]
    ns = st["gen_nonslack"]
    p_min = np.zeros(n)
    p_max = np.zeros(n)
    np.add.at(p_min, st["gen_bus"][ns], gen.min_p_mw.values[ns] / BASE_MVA)
    np.add.at(p_max, st["gen_bus"][ns], gen.max_p_mw.values[ns] / BASE_MVA)
    p_min = np.where(slack_np, 0.0, p_min)
    p_max = np.where(slack_np, 0.0, p_max)

    # Normalized BFS depth from the slack bus, PowerForm's `grid.depth` channel.
    f_np = grid.f_bus.numpy()
    t_np = grid.t_bus.numpy()
    adj: list[list[int]] = [[] for _ in range(n)]
    for fb, tb in zip(f_np.tolist(), t_np.tolist()):
        adj[fb].append(tb)
        adj[tb].append(fb)
    depth = np.full(n, np.inf)
    queue = [i for i in range(n) if slack_np[i]]
    for i in queue:
        depth[i] = 0.0
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        for j in adj[i]:
            if not np.isfinite(depth[j]):
                depth[j] = depth[i] + 1.0
                queue.append(j)
    depth = np.where(np.isfinite(depth), depth, float(n))
    depth = depth / max(depth.max(), 1.0)

    static = np.stack([
        grid.gs.numpy(), grid.bs.numpy(), vm_min, vm_max, p_min, p_max,
        slack_np.astype(np.float64), st["m_node"].numpy().astype(np.float64),
        depth,
    ], axis=-1)

    # Edge features: PowerForm's 8 line constants, in both directions.
    r = grid.br_r.numpy()
    x = grid.br_x.numpy()
    base = np.stack([
        np.log(np.clip(np.abs(r), 1e-6, None)),
        np.log(np.clip(np.abs(x), 1e-6, None)),
        np.sign(x),
        grid.br_b.numpy(),
        grid.tap.numpy(),
        grid.shift_rad.numpy(),
        np.clip(grid.s_rate.numpy(), None, 20.0),
        np.zeros_like(r),               # m_edge: no edge decisions here
    ], axis=-1)
    e = len(r)
    fwd = np.concatenate([base, np.ones((e, 1))], axis=-1)
    rev = np.concatenate([base, -np.ones((e, 1))], axis=-1)

    return {
        "static": torch.tensor(static, dtype=torch.float32, device=device),
        "p_min": torch.tensor(p_min, dtype=torch.float32, device=device),
        "p_max": torch.tensor(p_max, dtype=torch.float32, device=device),
        "vm_min": torch.tensor(vm_min, dtype=torch.float32, device=device),
        "vm_max": torch.tensor(vm_max, dtype=torch.float32, device=device),
        "src": torch.tensor(np.concatenate([f_np, t_np]), dtype=torch.long,
                            device=device),
        "dst": torch.tensor(np.concatenate([t_np, f_np]), dtype=torch.long,
                            device=device),
        "edge_feat": torch.tensor(np.concatenate([fwd, rev], axis=0),
                                  dtype=torch.float32, device=device),
        "m_free": st["m_free"].to(device),
        "n": n,
    }


def scenario_channels(lab: Dict[str, torch.Tensor], st: Dict) -> np.ndarray:
    """The 5 scenario-varying per-bus channels, (n, 5).

    Raw and standardized p/q, following PowerForm's `embed_s_const`. The fifth
    channel is the known slack voltage magnitude, 1.0 elsewhere.
    """
    p = lab["p_load"]
    q = lab["q_load"]
    v_sl = torch.ones_like(p)
    v_sl[st["slack"]] = lab["vm_ref"][st["slack"]]
    return torch.stack([p, q, standardize(p), standardize(q), v_sl],
                       dim=-1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Decision head: PowerForm's DecisionReadout on a batched [B, n, d]
# ---------------------------------------------------------------------------
class DecisionHead(nn.Module):
    """(P_g, V_set) read through a box sigmoid, one net per decision slot.

    Each slot's net sees the latent concatenated with its own box, and the
    sigmoid keeps the output inside the box. Non-decision buses emit exactly
    zero.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net_p = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))
        self.net_v = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, ctx: Dict, m_free: torch.Tensor):
        b = h.shape[0]

        def box(lo, hi):
            return (lo.expand(b, -1).unsqueeze(-1),
                    hi.expand(b, -1).unsqueeze(-1))

        p_lo, p_hi = box(ctx["p_min"], ctx["p_max"])
        v_lo, v_hi = box(ctx["vm_min"], ctx["vm_max"])
        p_g = (p_lo[..., 0] + (p_hi - p_lo)[..., 0]
               * torch.sigmoid(self.net_p(torch.cat([h, p_lo, p_hi], -1))[..., 0]))
        vm = (v_lo[..., 0] + (v_hi - v_lo)[..., 0]
              * torch.sigmoid(self.net_v(torch.cat([h, v_lo, v_hi], -1))[..., 0]))
        zero = torch.zeros_like(p_g)
        m = m_free.unsqueeze(0)
        return torch.where(m, p_g, zero), torch.where(m, vm, zero)
