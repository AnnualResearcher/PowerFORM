"""The in-distribution corpus: gridfm-datakit case500_goc.

30,000 IPOPT-labelled scenarios generated with a deliberately wide load band,
as the Table 1 trainers read them. There are two readers of the same solutions:

  * ``load_case500`` reads the datakit parquet directly, for the PowerForm rows;
  * ``build_static`` and ``scenario_labels`` read the static grid from the
    parquet and each scenario from its ``.pyg.json`` export, for the baseline
    rows, which consume OPFData-style JSON.

The parquet holds 29,987 scenarios (30,000 minus 13 failed solves). The
evaluation hold-out is the last 500 by ``load_scenario_idx``, ids 29,500-29,999;
everything before it, 29,487 scenarios, is training data.

The full corpus loads at about 13 GB per process. ``scenario_ids`` restricts a
load to the listed scenarios through parquet filter push-down.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from powerform.data.corpus.datakit_export import require_raw
from powerform.physics.newton_solver import TransmissionGrid

BASE_MVA = 100.0
N_HOLDOUT = 500
HOLDOUT_START = 29500
HOLDOUT_END = 29999

_F64 = torch.float64


def raw_dir_of(data_dir) -> Path:
    """``<data_dir>/case500_goc/raw``, the datakit export's parquet directory."""
    return Path(data_dir) / "case500_goc" / "raw"


# ---------------------------------------------------------------------------
# Parquet reader (PowerForm rows)
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    p_load: torch.Tensor       # (N,) pu
    q_load: torch.Tensor       # (N,) pu
    pg_bus: torch.Tensor       # (N,) pu label (bus-aggregated, non-slack gens)
    qg_bus: torch.Tensor       # (N,) pu label
    vm_ref: torch.Tensor       # (N,) pu solved OPF voltage magnitude
    va_ref: torch.Tensor       # (N,) rad solved angle
    v_slack: torch.Tensor      # (N,) complex slack voltage
    cost_ref: float            # true generation cost (from per-gen labels)
    sid: int = -1              # load_scenario_idx


@dataclass
class Case500Data:
    grid: TransmissionGrid       # static template with zero loads
    scenarios: list
    gen_bus: torch.Tensor        # (G,) bus idx of each gen unit (non-slack)
    m_node: torch.Tensor         # (N,) free-injection mask (non-slack gen buses)
    p_min: torch.Tensor          # (N,) bus box
    p_max: torch.Tensor
    vm_min: torch.Tensor         # (N,) bus voltage box
    vm_max: torch.Tensor
    scenario_ids: List[int] = field(default_factory=list)


def corpus_ids(data_dir) -> List[int]:
    """Every load_scenario_idx in the corpus, ascending, read from the bus table
    ``load_case500`` groups scenarios by."""
    require_raw(raw_dir_of(data_dir))
    col = pd.read_parquet(raw_dir_of(data_dir) / "bus_data.parquet",
                          columns=["load_scenario_idx"])["load_scenario_idx"]
    return [int(s) for s in np.unique(col.values)]


def table1_split(ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    """(train ids, hold-out ids) of the Table 1 corpus, checked two ways.

    The rule is positional, the last ``min(500, max(2, n // 5))`` scenarios. A
    corpus whose hold-out is not exactly ids 29,500-29,999, or that has a
    training id inside that range, is refused instead of silently shifting the
    evaluation set.
    """
    ids = sorted(int(i) for i in ids)
    n_hold = min(N_HOLDOUT, max(2, len(ids) // 5))
    train, hold = ids[:-n_hold], ids[-n_hold:]
    if hold != list(range(HOLDOUT_START, HOLDOUT_END + 1)):
        raise SystemExit(
            f"hold-out is not ids {HOLDOUT_START}..{HOLDOUT_END}: got "
            f"{len(hold)} ids {hold[0]}..{hold[-1]}")
    if train and max(train) >= HOLDOUT_START:
        raise SystemExit("a training id lies inside the hold-out range")
    return train, hold


def _read(raw: Path, name: str, ids: Optional[Sequence[int]]) -> pd.DataFrame:
    if ids is None:
        return pd.read_parquet(raw / name)
    return pd.read_parquet(
        raw / name,
        filters=[("load_scenario_idx", "in", [float(i) for i in ids])])


def load_case500(data_dir: str, scenario_ids: Optional[Sequence[int]] = None
                 ) -> Case500Data:
    """The datakit corpus as the PowerForm trainer reads it.

    ``scenario_ids`` = None loads every scenario. A list loads only those, in
    ascending id order. Either way the grid template comes from the corpus's
    first scenario, so a restricted load equals the matching slice of a full one.
    """
    raw = raw_dir_of(data_dir)
    require_raw(raw)
    if scenario_ids is None:
        bus = _read(raw, "bus_data.parquet", None)
        branch = _read(raw, "branch_data.parquet", None)
        gen = _read(raw, "gen_data.parquet", None)
        s0 = bus.load_scenario_idx.min()
        b0 = bus[bus.load_scenario_idx == s0].sort_values("bus")
        br0 = branch[branch.load_scenario_idx == s0].sort_values("idx")
        g0 = gen[gen.load_scenario_idx == s0].sort_values("idx")
    else:
        s0 = float(min(corpus_ids(data_dir)))
        want = sorted({int(i) for i in scenario_ids})
        bus = _read(raw, "bus_data.parquet", want)
        gen = _read(raw, "gen_data.parquet", want)
        b0 = _read(raw, "bus_data.parquet", [s0]).sort_values("bus")
        br0 = _read(raw, "branch_data.parquet", [s0]).sort_values("idx")
        g0 = _read(raw, "gen_data.parquet", [s0]).sort_values("idx")
        got = sorted(int(s) for s in np.unique(bus.load_scenario_idx.values))
        if got != want:
            missing = sorted(set(want) - set(got))
            raise SystemExit(f"{len(missing)} requested scenario(s) are not in "
                             f"the corpus, e.g. {missing[:5]}")
    br0 = br0[br0.br_status > 0]
    n = len(b0)

    slack_mask = torch.tensor(b0.REF.values.astype(bool))
    grid = TransmissionGrid(
        f_bus=torch.tensor(br0.from_bus.values, dtype=torch.int64),
        t_bus=torch.tensor(br0.to_bus.values, dtype=torch.int64),
        br_r=torch.tensor(br0.r.values, dtype=_F64),
        br_x=torch.tensor(br0.x.values, dtype=_F64),
        br_b=torch.tensor(br0.b.values, dtype=_F64),
        tap=torch.tensor(np.where(br0.tap.values == 0, 1.0, br0.tap.values),
                         dtype=_F64),
        shift_rad=torch.tensor(np.deg2rad(br0["shift"].values), dtype=_F64),
        # datakit's GS and BS columns are already per-unit
        gs=torch.tensor(b0.GS.values, dtype=_F64),
        bs=torch.tensor(b0.BS.values, dtype=_F64),
        p_load=torch.zeros(n, dtype=_F64), q_load=torch.zeros(n, dtype=_F64),
        is_slack=slack_mask,
    )

    # Static extras on the grid: line ratings (pu; 0 = unlimited -> sentinel)
    # and normalised BFS depth from the slack bus.
    rate = br0.rate_a.values / BASE_MVA
    rate = np.where(rate <= 0, 99.0, rate)
    grid.s_rate = torch.tensor(rate, dtype=_F64)
    adj = [[] for _ in range(n)]
    for fb, tb in zip(grid.f_bus.tolist(), grid.t_bus.tolist()):
        adj[fb].append(tb)
        adj[tb].append(fb)
    INF = float("inf")
    dep = [INF] * n
    queue = [i for i in range(n) if bool(slack_mask[i])]
    for i in queue:
        dep[i] = 0.0
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        for j in adj[i]:
            if dep[j] == INF:
                dep[j] = dep[i] + 1.0
                queue.append(j)
    dep_t = torch.tensor([d if d != INF else float(n) for d in dep],
                         dtype=_F64)
    grid.depth = dep_t / dep_t.max().clamp_min(1.0)

    # Slack units are not decision slots: the slack injection is determined by
    # the power flow.
    g0 = g0[g0.in_service > 0]
    unit_ok = ~g0.is_slack_gen.values.astype(bool)
    gen_bus = torch.tensor(g0.bus.values[unit_ok], dtype=torch.int64)
    gen_pmax = torch.tensor(g0.max_p_mw.values[unit_ok]) / BASE_MVA
    gen_pmin = torch.tensor(g0.min_p_mw.values[unit_ok]) / BASE_MVA

    # Bus-level boxes (sum of unit boxes per bus)
    m_node = torch.zeros(n, dtype=torch.bool)
    p_min = torch.zeros(n, dtype=_F64)
    p_max = torch.zeros(n, dtype=_F64)
    for gi in range(len(gen_bus)):
        b = int(gen_bus[gi])
        m_node[b] = True
        p_min[b] += float(gen_pmin[gi])
        p_max[b] += float(gen_pmax[gi])
    m_node = m_node & ~slack_mask

    def unit_cost(p_mw_units: np.ndarray, ok: np.ndarray, gdf) -> float:
        c2u = gdf.cp2_eur_per_mw2.values[ok]
        c1u = gdf.cp1_eur_per_mw.values[ok]
        c0u = gdf.cp0_eur.values[ok]
        p = p_mw_units[ok]
        return float((c2u * p ** 2 + c1u * p + c0u).sum())

    scenarios = []
    for sid, bs_ in bus.groupby("load_scenario_idx"):
        bsc = bs_.sort_values("bus")
        gsc = gen[gen.load_scenario_idx == sid].sort_values("idx")
        gsc = gsc[gsc.in_service > 0]
        ok = ~gsc.is_slack_gen.values.astype(bool)
        pg_bus = torch.zeros(n, dtype=_F64)
        qg_bus = torch.zeros(n, dtype=_F64)
        gb = gsc.bus.values[ok]
        for b, p_mw, q_mv in zip(gb, gsc.p_mw.values[ok],
                                 gsc.q_mvar.values[ok]):
            pg_bus[int(b)] += p_mw / BASE_MVA
            qg_bus[int(b)] += q_mv / BASE_MVA
        vm = torch.tensor(bsc.Vm.values, dtype=_F64)
        va = torch.tensor(np.deg2rad(bsc.Va.values), dtype=_F64)
        v_sl = torch.ones(n, dtype=torch.complex128)
        v_sl[slack_mask] = (vm[slack_mask]
                            * torch.exp(1j * va[slack_mask]))
        cost_ref = unit_cost(gsc.p_mw.values, np.ones(len(gsc), bool), gsc)
        scenarios.append(Scenario(
            p_load=torch.tensor(bsc.Pd.values / BASE_MVA),
            q_load=torch.tensor(bsc.Qd.values / BASE_MVA),
            pg_bus=pg_bus, qg_bus=qg_bus,
            vm_ref=vm, va_ref=va, v_slack=v_sl, cost_ref=cost_ref,
            sid=int(sid),
        ))

    vm_min = torch.tensor(b0.min_vm_pu.values, dtype=_F64)
    vm_max = torch.tensor(b0.max_vm_pu.values, dtype=_F64)
    return Case500Data(grid=grid, scenarios=scenarios, gen_bus=gen_bus,
                       m_node=m_node, p_min=p_min, p_max=p_max,
                       vm_min=vm_min, vm_max=vm_max,
                       scenario_ids=[s.sid for s in scenarios])


def template_gen_df(data_dir, s0: Optional[float] = None) -> pd.DataFrame:
    """The generator table of the corpus's first scenario. Costs are fixed across
    the corpus, so the PowerForm trainer scores cost with this template."""
    raw = raw_dir_of(data_dir)
    require_raw(raw)
    if s0 is None:
        s0 = float(min(corpus_ids(data_dir)))
    return _read(raw, "gen_data.parquet", [s0])


class TransmissionInstance:
    """The model's view of one scenario on the static grid, labels included, with
    every tensor on ``device``."""

    def __init__(self, grid: TransmissionGrid, scen: Scenario,
                 device="cpu"):
        d = device
        self.tgrid = TransmissionGrid(
            f_bus=grid.f_bus.to(d), t_bus=grid.t_bus.to(d),
            br_r=grid.br_r.to(d), br_x=grid.br_x.to(d),
            br_b=grid.br_b.to(d), tap=grid.tap.to(d),
            shift_rad=grid.shift_rad.to(d), gs=grid.gs.to(d),
            bs=grid.bs.to(d),
            p_load=scen.p_load.to(d), q_load=scen.q_load.to(d),
            is_slack=grid.is_slack.to(d),
        )
        self.edge_uv = torch.stack([self.tgrid.f_bus, self.tgrid.t_bus],
                                   dim=1)
        self.edge_r = self.tgrid.br_r.clamp_min(1e-6)
        self.edge_x = self.tgrid.br_x.abs().clamp_min(1e-6)
        self.p_load = self.tgrid.p_load
        self.q_load = self.tgrid.q_load
        self.is_sub = self.tgrid.is_slack
        self.pg_label = scen.pg_bus.to(d)
        self.qg_label = scen.qg_bus.to(d)
        self.vm_ref = scen.vm_ref.to(d)
        self.va_ref = scen.va_ref.to(d)
        self.v_slack_scalar = scen.v_slack.abs().to(d)
        self.s_max = grid.s_rate.to(d)
        self.depth = grid.depth.to(d)

    @property
    def n_nodes(self) -> int:
        return self.tgrid.n_nodes

    @property
    def n_edges(self) -> int:
        return self.tgrid.n_edges


# ---------------------------------------------------------------------------
# The corpus's cost convention
# ---------------------------------------------------------------------------
def equal_marginal_alloc(P_bus: float, c2, c1, pmin, pmax) -> np.ndarray:
    """Allocate bus-level P among units minimizing quadratic cost: water-filling
    on marginal cost with box clipping, by bisection on lambda."""
    c2 = np.maximum(c2, 1e-9)

    def total(lmbda):
        p = (lmbda - c1) / (2 * c2)
        return np.clip(p, pmin, pmax).sum()

    lo, hi = -1e6, 1e6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(mid) < P_bus:
            lo = mid
        else:
            hi = mid
    p = np.clip((0.5 * (lo + hi) - c1) / (2 * c2), pmin, pmax)
    # a proportional nudge within the bounds removes the residual mismatch
    diff = P_bus - p.sum()
    room = (pmax - p) if diff > 0 else (p - pmin)
    if room.sum() > 1e-12:
        p = p + np.sign(diff) * room * (abs(diff) / room.sum())
    return p


def slack_quadratic(gen_df):
    """Effective quadratic cost of the slack bus units (pu basis)."""
    g = gen_df[gen_df.in_service > 0]
    sl = g[g.is_slack_gen.astype(bool)]
    if not len(sl):
        return (0.0, 0.0, 0.0)
    c2 = np.maximum(sl.cp2_eur_per_mw2.values, 1e-6)
    c1 = sl.cp1_eur_per_mw.values
    c0 = sl.cp0_eur.values

    def cost_at(P_mw):
        s2 = (1.0 / (2 * c2)).sum()
        lam = (P_mw + (c1 / (2 * c2)).sum()) / s2
        p = (lam - c1) / (2 * c2)
        return float((c2 * p ** 2 + c1 * p + c0).sum())

    y0, y1, y2 = cost_at(0.0), cost_at(100.0), cost_at(200.0)
    a2 = (y2 - 2 * y1 + y0) / (2 * 100.0 ** 2)
    a1 = (y1 - y0) / 100.0 - a2 * 100.0
    return (a2 * BASE_MVA ** 2, a1 * BASE_MVA, y0)


def dispatch_cost(pg_bus_pu: np.ndarray, p_slack_pu: float, gen_bus_np,
                  ns: pd.DataFrame, slack_coef: Tuple[float, float, float]
                  ) -> float:
    """Predicted generation cost of a bus-level dispatch (PowerForm rows).

    Non-slack buses split their dispatch among units by equal-marginal
    allocation; the slack's cost is its quadratic fit (``slack_quadratic``)
    evaluated at the terminal power flow's slack injection.
    """
    s2c, s1c, s0c = slack_coef
    cost = 0.0
    pg = pg_bus_pu * BASE_MVA
    for b in np.unique(gen_bus_np):
        sub = ns[ns.bus == b]
        alloc = equal_marginal_alloc(
            pg[int(b)],
            np.maximum(sub.cp2_eur_per_mw2.values, 1e-9),
            sub.cp1_eur_per_mw.values,
            sub.min_p_mw.values, sub.max_p_mw.values)
        cost += float((sub.cp2_eur_per_mw2.values * alloc ** 2
                       + sub.cp1_eur_per_mw.values * alloc
                       + sub.cp0_eur.values).sum())
    p_sl = float(p_slack_pu)
    cost += s2c * p_sl ** 2 + s1c * p_sl + s0c
    return cost


def predicted_cost(pgn_mw: np.ndarray, p_slack_mw: float, st,
                   ns_df, sl_df) -> tuple:
    """Predicted generation cost from a bus-level dispatch (baseline rows).

    Returns ``(c_nonslack, c_slack)``. The non-slack part splits each bus's
    dispatch by equal-marginal allocation; the slack part evaluates the slack
    units at the terminal power flow's ``p_slack``, shared by box width when
    there is more than one. The slack carries about 3.1% of total cost on
    case500_goc, so a score must include both.
    """
    c = 0.0
    for b in np.unique(st["gen_bus"][st["gen_nonslack"]]):
        sub = ns_df[ns_df.bus == b]
        a = equal_marginal_alloc(
            pgn_mw[int(b)],
            np.maximum(sub.cp2_eur_per_mw2.values, 1e-9),
            sub.cp1_eur_per_mw.values,
            sub.min_p_mw.values, sub.max_p_mw.values)
        c += float((sub.cp2_eur_per_mw2.values * a ** 2
                    + sub.cp1_eur_per_mw.values * a
                    + sub.cp0_eur.values).sum())

    pr = (sl_df.max_p_mw.values - sl_df.min_p_mw.values).clip(1e-9)
    psh = pr / pr.sum()
    c_sl = float((sl_df.cp2_eur_per_mw2.values * (p_slack_mw * psh) ** 2
                  + sl_df.cp1_eur_per_mw.values * (p_slack_mw * psh)
                  + sl_df.cp0_eur.values).sum())
    return c, c_sl


# ---------------------------------------------------------------------------
# .pyg.json reader (baseline rows)
# ---------------------------------------------------------------------------
def build_static(raw_dir) -> Dict:
    """Scenario-invariant grid, bus masks and generator indexing, read from the
    first hold-out scenario; the topology is the same in every scenario."""
    raw = Path(raw_dir)
    require_raw(raw)
    s0 = float(HOLDOUT_START)
    bus = (pd.read_parquet(raw / "bus_data.parquet",
                           filters=[("load_scenario_idx", "==", s0)])
           .sort_values("bus").reset_index(drop=True))
    br = pd.read_parquet(raw / "branch_data.parquet",
                         filters=[("load_scenario_idx", "==", s0)]).sort_values("idx")
    br = br[br.br_status > 0].reset_index(drop=True)
    gen = (pd.read_parquet(raw / "gen_data.parquet",
                           filters=[("load_scenario_idx", "==", s0)])
           .sort_values("idx"))
    gen = gen[gen.in_service > 0].reset_index(drop=True)
    n = len(bus)

    slack = torch.tensor(bus.REF.values.astype(bool))
    grid = TransmissionGrid(
        f_bus=torch.tensor(br.from_bus.values, dtype=torch.int64),
        t_bus=torch.tensor(br.to_bus.values, dtype=torch.int64),
        br_r=torch.tensor(br.r.values, dtype=_F64),
        br_x=torch.tensor(br.x.values, dtype=_F64),
        br_b=torch.tensor(br.b.values, dtype=_F64),
        tap=torch.tensor(np.where(br.tap.values == 0, 1.0, br.tap.values),
                         dtype=_F64),
        shift_rad=torch.tensor(np.deg2rad(br["shift"].values), dtype=_F64),
        gs=torch.tensor(bus.GS.values, dtype=_F64),
        bs=torch.tensor(bus.BS.values, dtype=_F64),
        p_load=torch.zeros(n, dtype=_F64),
        q_load=torch.zeros(n, dtype=_F64),
        is_slack=slack,
    )
    rate = br.rate_a.values / BASE_MVA
    grid.s_rate = torch.tensor(np.where(rate <= 0, 99.0, rate), dtype=_F64)

    ns = ~gen.is_slack_gen.values.astype(bool)
    gen_bus = gen.bus.values.astype(int)
    m_node = torch.zeros(n, dtype=torch.bool)
    m_node[gen_bus[ns]] = True
    m_node &= ~slack
    m_free = m_node | slack
    return dict(grid=grid, n=n, slack=slack, m_node=m_node, m_free=m_free,
                gen_bus=gen_bus, gen_nonslack=ns, gen_df=gen)


def scenario_labels(path: Path, st: Dict) -> Dict[str, torch.Tensor]:
    """Loads and labels for one scenario, from the exported ``.pyg.json``."""
    o = json.loads(Path(path).read_text())
    n = st["n"]
    lb = np.asarray(o["metadata"]["load_bus_map"], dtype=int) - 1
    ld = np.asarray(o["grid"]["nodes"]["load"], dtype=float)
    p_load = np.zeros(n)
    q_load = np.zeros(n)
    np.add.at(p_load, lb, ld[:, 0])
    np.add.at(q_load, lb, ld[:, 1])

    sol_bus = np.asarray(o["solution"]["nodes"]["bus"], dtype=float)
    sol_gen = np.asarray(o["solution"]["nodes"]["generator"], dtype=float)
    ns = st["gen_nonslack"]
    pg_label = np.zeros(n)
    np.add.at(pg_label, st["gen_bus"][ns], sol_gen[ns, 0])
    qg_label = np.zeros(n)
    np.add.at(qg_label, st["gen_bus"][ns], sol_gen[ns, 1])

    return {
        "p_load": torch.tensor(p_load), "q_load": torch.tensor(q_load),
        "vm_ref": torch.tensor(sol_bus[:, 1]), "va_ref": torch.tensor(sol_bus[:, 0]),
        "pg_label": torch.tensor(pg_label), "qg_label": torch.tensor(qg_label),
        "cost_ref": float(o["metadata"]["objective"]),
    }


def pyg_split(pyg_root) -> Tuple[List[Path], List[Path]]:
    """(train paths, hold-out paths) of the ``.pyg.json`` export, checked.

    ``pyg_root`` holds ``train_opfdata/`` and ``holdout_opfdata/``, one
    ``scenario_<id>.pyg.json`` per scenario. The hold-out must be exactly ids
    29,500-29,999 and no training file may carry an id at or above 29,500.
    """
    root = Path(pyg_root)
    if not (root / "holdout_opfdata").is_dir():
        raise SystemExit(
            f"the converted Table 1 corpus is not at {root}; convert it as step 4 "
            f"of powerform/data/corpus/README.md describes, or point POWERFORM_DATA "
            f"at the data home that holds it")
    train = sorted((root / "train_opfdata").glob("*.pyg.json"))
    hold = sorted((root / "holdout_opfdata").glob("*.pyg.json"))

    def sid(p: Path) -> int:
        return int(p.name.split("_")[1].split(".")[0])

    hold_ids = [sid(p) for p in hold]
    if hold_ids != list(range(HOLDOUT_START, HOLDOUT_END + 1)):
        raise SystemExit(f"{root / 'holdout_opfdata'} is not ids "
                         f"{HOLDOUT_START}..{HOLDOUT_END} ({len(hold)} files)")
    if any(sid(p) >= HOLDOUT_START for p in train):
        raise SystemExit(f"{root / 'train_opfdata'} contains hold-out ids")
    return train, hold
