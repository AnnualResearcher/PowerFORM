"""Operating-limit violations of a voltage state, under one convention.

`audit_state` takes a voltage state and a scenario's static data and returns the
four inequality axes:

    V box        vmin <= vm <= vmax                        (per bus)
    derived Q    Im(V conj(YV)) + Qd  in [sum Qmin, sum Qmax] at free buses
    derived P    Re(V conj(YV)) + Pd  in [sum Pmin, sum Pmax] at free buses
    thermal      |S_from| / rate_a <= 1                     (per branch)

Every axis is a function of the state and the grid, not of what a model claims to
dispatch, so a row that solves the power flow and a row that regresses the state
go in one column. The convention: tolerance 1e-6 pu; free buses are in-service
generator buses and the slack; generator limits are summed over in-service units
at the bus; counts are per-scenario means of violating elements; "worst" is the
maximum over the run.

The Table 1 cell is `violated_inequalities_per_scenario` over the 500 hold-out
scenarios of the case500_goc corpus. To score another model, pass any iterable of
`(sid, vm, va)`, one per hold-out scenario, to `audit_states`.

    python -m powerform.eval.violations --ckpt checkpoints/case500_physics_seed0.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from powerform.data.corpus.datakit_export import CORPUS_DIR
from powerform.data.datakit_loader import (BASE_MVA, HOLDOUT_START, N_HOLDOUT,
                                           TransmissionInstance, corpus_ids,
                                           load_case500, raw_dir_of, table1_split)
from powerform.device import resolve_device
from powerform.eval.checkpoint import load_model
from powerform.model.spec import TaskSpec
from powerform.paths import OUTPUTS, shown
from powerform.physics.newton_solver import (TransmissionGrid, build_ybus,
                                             transmission_pf_newton_pv)

TOL = 1e-6                      # pu for voltage and power, relative for loading
OUT_DIR = OUTPUTS / "table1"


def build_grid(raw_dir: Path) -> tuple:
    """Static grid, Ybus and branch ratings of case500_goc, read from the first
    hold-out scenario; the topology is the same in every scenario."""
    raw = Path(raw_dir)
    s0 = HOLDOUT_START
    bus = pd.read_parquet(raw / "bus_data.parquet",
                          filters=[("load_scenario_idx", "==", float(s0))]
                          ).sort_values("bus").reset_index(drop=True)
    br = pd.read_parquet(raw / "branch_data.parquet",
                         filters=[("load_scenario_idx", "==", float(s0))]
                         ).sort_values("idx")
    br = br[br.br_status > 0].reset_index(drop=True)
    n = len(bus)
    g = TransmissionGrid(
        f_bus=torch.tensor(br.from_bus.values, dtype=torch.int64),
        t_bus=torch.tensor(br.to_bus.values, dtype=torch.int64),
        br_r=torch.tensor(br.r.values, dtype=torch.float64),
        br_x=torch.tensor(br.x.values, dtype=torch.float64),
        br_b=torch.tensor(br.b.values, dtype=torch.float64),
        tap=torch.tensor(np.where(br.tap.values == 0, 1.0, br.tap.values),
                         dtype=torch.float64),
        shift_rad=torch.tensor(np.deg2rad(br["shift"].values), dtype=torch.float64),
        gs=torch.tensor(bus.GS.values, dtype=torch.float64),    # already pu
        bs=torch.tensor(bus.BS.values, dtype=torch.float64),
        p_load=torch.zeros(n, dtype=torch.float64),
        q_load=torch.zeros(n, dtype=torch.float64),
        is_slack=torch.tensor(bus.REF.values.astype(bool)),
    )
    rate = br.rate_a.values / BASE_MVA
    return g, build_ybus(g).numpy(), np.where(rate <= 0, np.inf, rate), n


def scenario_statics(raw_dir: Path, n: int, slack: np.ndarray,
                     sids: Optional[Iterable[int]] = None) -> Dict[int, Dict]:
    """Loads, boxes and the free-bus mask per scenario, from the raw parquet;
    `sids` defaults to the hold-out."""
    raw = Path(raw_dir)
    sids = (np.arange(HOLDOUT_START, HOLDOUT_START + N_HOLDOUT) if sids is None
            else np.asarray(list(sids)))
    flt = [("load_scenario_idx", "in", [float(s) for s in sids])]
    bus_g = dict(tuple(pd.read_parquet(raw / "bus_data.parquet", filters=flt)
                       .groupby("load_scenario_idx")))
    gen_g = dict(tuple(pd.read_parquet(raw / "gen_data.parquet", filters=flt)
                       .groupby("load_scenario_idx")))

    out: Dict[int, Dict] = {}
    for sid in sids:
        b = bus_g[float(sid)].sort_values("bus")
        gs = gen_g[float(sid)].sort_values("idx")
        gs = gs[gs.in_service > 0]
        gbus = gs.bus.values.astype(int)

        free = np.zeros(n, dtype=bool)
        free[gbus] = True
        free |= slack

        def to_bus(vals: np.ndarray) -> np.ndarray:
            acc = np.zeros(n)
            np.add.at(acc, gbus, vals)
            return acc

        out[int(sid)] = {
            "pd": b.Pd.values / BASE_MVA, "qd": b.Qd.values / BASE_MVA,
            "vmin": b.min_vm_pu.values, "vmax": b.max_vm_pu.values,
            "vm_label": b.Vm.values, "va_label": np.deg2rad(b.Va.values),
            "free": free,
            "qmin": to_bus(gs.min_q_mvar.values / BASE_MVA),
            "qmax": to_bus(gs.max_q_mvar.values / BASE_MVA),
            "pmin": to_bus(gs.min_p_mw.values / BASE_MVA),
            "pmax": to_bus(gs.max_p_mw.values / BASE_MVA),
        }
    return out


def thermal(rate: np.ndarray, vm: np.ndarray, va: np.ndarray,
            grid: TransmissionGrid) -> np.ndarray:
    """Per-branch loading |S_from| / rate_a implied by (vm, va), with the pi model
    and the off-nominal tap on the from side."""
    v = vm * np.exp(1j * va)
    f, t = grid.f_bus.cpu().numpy(), grid.t_bus.cpu().numpy()
    y = 1.0 / (grid.br_r.cpu().numpy() + 1j * grid.br_x.cpu().numpy())
    bc2 = 1j * grid.br_b.cpu().numpy() / 2.0
    tap, sh = grid.tap.cpu().numpy(), grid.shift_rad.cpu().numpy()
    tt = tap * np.exp(1j * sh)
    vi, vj = v[f], v[t]
    i_ij = (y + bc2) * vi / (tap ** 2) - y * vj / np.conj(tt)
    s_ij = vi * np.conj(i_ij)
    return np.abs(s_ij) / rate


def _box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Two-sided box excess, >= 0 elementwise."""
    return np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)


def audit_state(vm: np.ndarray, va: np.ndarray, st: Dict, grid, Y, rate) -> Dict:
    """Every inequality axis for one scenario's voltage state."""
    free = st["free"]
    v = vm * np.exp(1j * va)
    s = v * np.conj(Y @ v)
    qg_der = s.imag + st["qd"]
    pg_der = s.real + st["pd"]

    vv = _box(vm, st["vmin"], st["vmax"])
    qv = np.where(free, _box(qg_der, st["qmin"], st["qmax"]), 0.0)
    pv_ = np.where(free, _box(pg_der, st["pmin"], st["pmax"]), 0.0)
    ld = thermal(rate, vm, va, grid)
    slack = grid.is_slack.cpu().numpy()
    return {
        "v_n": int((vv > TOL).sum()),
        "v_max": float(vv.max()),
        "q_n": int((qv > TOL).sum()),
        "q_max": float(qv.max()),
        "p_n": int((pv_ > TOL).sum()),
        "p_max": float(pv_.max()),
        "p_n_slack": int((pv_[slack] > TOL).sum()),
        "p_n_nonslack": int((pv_[free & ~slack] > TOL).sum()),
        "p_max_slack": float(pv_[slack].max()),
        "load_max": float(ld.max()),
        "n_over": int((ld > 1.0 + TOL).sum()),
    }


def aggregate(recs: List[Dict]) -> Dict:
    """Per-scenario records -> the row's numbers. The derived-P axis enters the
    total as the slack bus only: its non-slack part is zero on a solved state."""
    a = {k: np.asarray([r[k] for r in recs], dtype=float) for k in recs[0]}

    def axis(n_key: str, max_key: str) -> Dict:
        return {"per_scenario_mean": float(a[n_key].mean()),
                "scenarios_with_any": int((a[n_key] > 0).sum()),
                "worst": float(a[max_key].max())}

    total = a["v_n"] + a["q_n"] + a["n_over"] + a["p_n_slack"]
    return {
        "state": "realised",
        "n_scenarios": len(recs),
        "voltage_box": axis("v_n", "v_max"),
        "derived_Q": axis("q_n", "q_max"),
        "derived_P": {**axis("p_n", "p_max"),
                      "slack_bus_scenarios": int((a["p_n_slack"] > 0).sum()),
                      "nonslack_per_scenario_mean": float(a["p_n_nonslack"].mean()),
                      "slack_worst": float(a["p_max_slack"].max())},
        "thermal": {"max_loading": float(a["load_max"].max()),
                    "overloads_per_scenario": float(a["n_over"].mean()),
                    "scenarios_with_overload": int((a["n_over"] > 0).sum())},
        "violated_inequalities_per_scenario": float(total.mean()),
        "scenarios_with_any": int((total > 0).sum()),
    }


def audit_states(states: Iterable[Tuple], statics: Dict, grid, Y, rate,
                 expect_n: Optional[int] = N_HOLDOUT) -> Dict:
    """Audit every `(sid, vm, va)` and aggregate the row. A duplicated scenario, or
    a count other than `expect_n`, stops the row: an average over the wrong set of
    scenarios looks exactly like a correct one."""
    recs: List[Dict] = []
    seen = set()
    for sid, vm, va in states:
        if sid in seen:
            raise SystemExit(f"duplicate scenario {sid}")
        seen.add(sid)
        recs.append(audit_state(vm, va, statics[sid], grid, Y, rate))
        if len(recs) % 100 == 0:
            print(f"  {len(recs)}/{expect_n or '?'}", flush=True)
    if expect_n is not None and len(recs) != expect_n:
        raise SystemExit(f"audited {len(recs)} scenarios, expected {expect_n}")
    return aggregate(recs)


def powerform_states(model, data, device, limit: Optional[int] = None) -> Iterable[Tuple]:
    """PowerForm -> (p_g, vm_set) -> terminal PV solve, per hold-out scenario.

    `data` holds the hold-out through `load_case500`. The 0.90/1.10 setpoint clamp
    belongs to this corpus and every row of its table was scored with it. `limit`
    stops after that many scenarios.
    """
    hold = data.scenarios[-N_HOLDOUT:]
    slack_mask = data.grid.is_slack
    m_free = data.m_node | slack_mask
    p_min = torch.where(slack_mask, torch.zeros_like(data.p_min), data.p_min)
    p_max = torch.where(slack_mask, torch.zeros_like(data.p_max), data.p_max)
    spec = TaskSpec(m_node=m_free.to(device),
                    m_edge=torch.zeros(data.grid.n_edges, dtype=torch.bool,
                                       device=device),
                    p_min=p_min.to(device), p_max=p_max.to(device),
                    vm_min=data.vm_min.to(device), vm_max=data.vm_max.to(device))
    with torch.no_grad():
        for si, scen in enumerate(hold):
            if limit is not None and si >= limit:
                return
            if scen.sid != HOLDOUT_START + si:
                raise SystemExit(f"hold-out position {si} holds scenario "
                                 f"{scen.sid}, not {HOLDOUT_START + si}")
            inst = TransmissionInstance(data.grid, scen, device=device)
            out = model(inst, spec)
            vm_set = torch.where(m_free.to(device),
                                 out["vm_set"].clamp(0.90, 1.10),
                                 torch.ones_like(out["vm_set"]))
            pv = m_free.to(device) & ~inst.tgrid.is_slack
            pf = transmission_pf_newton_pv(inst.tgrid, out["p_g"], vm_set, pv,
                                           n_iter=40)
            V = pf["V"].cpu()
            yield scen.sid, V.abs().numpy(), torch.angle(V).numpy()


def check_alignment(data, statics: Dict) -> None:
    """Refuse to audit if the loader's hold-out is not the statics' hold-out: a
    shift would audit every state against its neighbour's loads."""
    hold = data.scenarios[-N_HOLDOUT:]
    for si in (0, len(hold) - 1):
        sid = HOLDOUT_START + si
        if not np.array_equal(hold[si].p_load.detach().cpu().numpy(),
                              statics[sid]["pd"]):
            raise SystemExit(f"hold-out misaligned at scenario {sid}: the "
                             "loader's loads are not the raw parquet's")


def load_holdout():
    """The 500 hold-out scenarios through the Table 1 loader, re-derived from the
    corpus's own id list. Loading only these keeps the process small."""
    _, hold = table1_split(corpus_ids(CORPUS_DIR))
    return load_case500(str(CORPUS_DIR), scenario_ids=hold)


def powerform_row_model(ck: Path, device):
    """A PowerForm checkpoint of this corpus. Its costs never vary, so such a model
    has no cost features; the baselines' rows ship as records."""
    try:
        model = load_model(ck, device)
    except ValueError as e:
        raise SystemExit(f"{ck.name} is not a PowerForm checkpoint, and only those are "
                         "audited here; the baseline rows of this column ship as records "
                         "in results/table1/violations.json") from e
    if getattr(model, "n_cost_feats", None) != 0:
        raise SystemExit(f"{ck.name} was not trained on the case500_goc corpus")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit case500 checkpoints on the hold-out (Table 1, viol./scen.).")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="PowerForm checkpoint trained on the case500_goc corpus; "
                         "repeat for several rows")
    ap.add_argument("--n", type=int, default=0,
                    help="first n hold-out scenarios only; 0 = all 500")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    raw_dir = raw_dir_of(CORPUS_DIR)
    if not (raw_dir / "bus_data.parquet").exists():
        raise SystemExit(f"the case500_goc corpus is not at {raw_dir}; "
                         "powerform/data/corpus/README.md says how to build it, or "
                         "point POWERFORM_DATA at a data home that holds it")
    grid, Y, rate, n = build_grid(raw_dir)
    statics = scenario_statics(raw_dir, n, grid.is_slack.numpy())
    n_free = int(statics[HOLDOUT_START]["free"].sum())
    print(f"grid {n} buses / {grid.f_bus.numel()} branches; free buses {n_free}; "
          f"tolerance {TOL} pu", flush=True)
    data = load_holdout()
    check_alignment(data, statics)

    limit = args.n or None
    # A subset has a file of its own, so it cannot replace a full row.
    out_path = Path(args.out_dir) / (f"violations_first{limit}.json" if limit
                                     else "violations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = (json.loads(out_path.read_text()) if out_path.exists() else
           {"tolerance_pu": TOL, "n_scenarios": limit or N_HOLDOUT, "n_free_buses": n_free,
            "rows": {}})
    for ck in map(Path, args.ckpt):
        print(f"\n{ck.name}", flush=True)
        model = powerform_row_model(ck, device)
        row = audit_states(powerform_states(model, data, device, limit=limit),
                           statics, grid, Y, rate,
                           expect_n=N_HOLDOUT if limit is None else limit)
        print(f"  violated inequalities per scenario "
              f"{row['violated_inequalities_per_scenario']:.4f}", flush=True)
        doc["rows"][ck.stem.removeprefix("case500_")] = row
        out_path.write_text(json.dumps(doc, indent=2))
    print(f"-> {shown(out_path)}", flush=True)


if __name__ == "__main__":
    main()
