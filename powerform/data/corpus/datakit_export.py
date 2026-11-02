"""gridfm-datakit case500_goc corpus -> one OPFData-shaped file per scenario.

The in-distribution corpus (Table 1, App. B) is 30,000 ``case500_goc`` load
scenarios generated and IPOPT-solved by gridfm-datakit
(``datakit/config_case500_30k.yaml``). datakit writes partitioned parquet tables,
and the baselines read OPFData's scenario JSON. This script writes each solved
scenario as one ``scenario_<id>.pyg.json`` in OPFData's structure: grid, solution,
metadata.

Two disjoint splits, selected with ``--split``:

  ``holdout``  the last 500 by ``load_scenario_idx`` (29500..29999), the
               evaluation set (load factors 0.84-1.05x nominal).
  ``train``    everything below 29500: 29,487 scenarios, because ids
               16496..16582 have 13 gaps where the OPF did not converge
               (datakit's ``error.log`` lists them).

Every scenario-invariant field built from the parquet is asserted identical to
an OPFData scenario of the same pglib case before anything is written
(``verify_against_sample``); the reference is the first line of OPFData's
``case500_goc`` fulltop train shard unless ``--reference`` names another.

Only three generator columns come from the reference rather than the parquet:
mbase and the pglib initial pg/qg setpoints (cols 0/1/4). datakit does not export
them; they are case-file constants and carry no per-scenario label information.

Usage (from the repository root; needs only numpy, pandas and pyarrow):
    python -m powerform.data.corpus.datakit_export --split holdout
    python -m powerform.data.corpus.datakit_export --split train
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from powerform.paths import DATA_HOME

BASE_MVA = 100.0
N_HOLDOUT = 500
HOLDOUT_START = 29500  # first load_scenario_idx of the evaluation split
# 29,500 ids below the holdout minus the 13 OPF solves that did not converge.
N_TRAIN = 29487

CHECKSUMS = Path(__file__).resolve().parent / "checksums.json"

DATAKIT_HOME = DATA_HOME / "case500_datakit"
CORPUS_DIR = DATAKIT_HOME / "case500_30k"
RAW_DIR = CORPUS_DIR / "case500_goc" / "raw"
REFERENCE_PATH = (DATA_HOME / "opfdata_hf" / "pglib_opf_case500_goc"
                  / "fulltop" / "train.jsonl.gz")
OUT_DIR = DATAKIT_HOME / "holdout_opfdata"
TRAIN_OUT_DIR = DATAKIT_HOME / "train_opfdata"

# Bus type codes used by the OPFData bus feature column BUS_TYPE_IDX (=1).
BUS_TYPE_PQ, BUS_TYPE_PV, BUS_TYPE_REF = 1, 2, 3


@dataclass(frozen=True)
class StaticGrid:
    """Scenario-invariant topology and parameters, in OPFData units."""

    bus: np.ndarray               # (N, 4) [base_kv, type, vmin, vmax]
    gen_static: np.ndarray        # (G, 11) with per-scenario cols already final
    gen_bus: np.ndarray           # (G,) 0-based bus index of each in-service unit
    gen_row_idx: np.ndarray       # (G,) datakit `idx` of each in-service unit
    gen_is_slack: np.ndarray      # (G,) bool
    load_bus: np.ndarray          # (L,) 0-based bus index of each load node
    shunt_bus: np.ndarray         # (S,) 0-based bus index of each shunt node
    shunt: np.ndarray             # (S, 2) [bs, gs] (already per-unit)
    ac_senders: np.ndarray
    ac_receivers: np.ndarray
    ac_feats: np.ndarray          # (E_ac, 9)
    ac_row_idx: np.ndarray        # (E_ac,) datakit branch `idx`, edge order
    tr_senders: np.ndarray
    tr_receivers: np.ndarray
    tr_feats: np.ndarray          # (E_tr, 11)
    tr_row_idx: np.ndarray        # (E_tr,) datakit branch `idx`, edge order


def require_raw(raw_dir: Path) -> None:
    """Refuse a missing corpus with where to get it, not a parquet traceback."""
    if not (Path(raw_dir) / "bus_data.parquet").exists():
        raise SystemExit(
            f"the Table 1 corpus is not at {raw_dir}; generate it as step 4 of "
            f"powerform/data/corpus/README.md describes, or point POWERFORM_DATA "
            f"at the data home that holds it")


def read_reference(path: Path) -> dict:
    """One OPFData scenario of the same pglib case, the structural anchor.

    ``.jsonl.gz`` (a consolidated shard): its first line. ``.json.gz``: the
    whole file. Anything else: plain JSON.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{path} missing; the structural reference is an OPFData "
            f"case500_goc scenario -- fetch it with python -m "
            f"powerform.data.corpus.fetch_opfdata --case pglib_opf_case500_goc "
            f"--variant fulltop, or pass --reference")
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.loads(fh.readline())
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def _bus_type(bus_df: pd.DataFrame) -> np.ndarray:
    return np.where(
        bus_df["REF"].values == 1,
        BUS_TYPE_REF,
        np.where(bus_df["PV"].values == 1, BUS_TYPE_PV, BUS_TYPE_PQ),
    ).astype(float)


def build_static_grid(bus0: pd.DataFrame, gen0: pd.DataFrame,
                      branch0: pd.DataFrame, sample: dict) -> StaticGrid:
    """Build the scenario-invariant grid from the first scenario's parquet rows.

    `sample` supplies only the two pglib initial-setpoint generator columns.
    """
    bus = np.stack([
        bus0["vn_kv"].values.astype(float),
        _bus_type(bus0),
        bus0["min_vm_pu"].values.astype(float),
        bus0["max_vm_pu"].values.astype(float),
    ], axis=1)

    # --- generators: keep in-service units only, ordered by datakit idx -----
    gin = gen0[gen0["in_service"] > 0]
    g_sample = np.asarray(sample["grid"]["nodes"]["generator"], dtype=float)
    if len(gin) != len(g_sample):
        raise ValueError(
            f"in-service generator count {len(gin)} != sample generator count "
            f"{len(g_sample)}; the sample is not the same case variant."
        )
    gen_static = np.zeros((len(gin), 11), dtype=float)
    # cols 0/1/4 are case-file constants datakit does not export; see the module
    # docstring. mbase agrees with max_p_mw to <0.21% (pglib rounds it), which is
    # the check below.
    mbase_rel = np.abs(g_sample[:, 0] - gin["max_p_mw"].values) / gin["max_p_mw"].values
    if mbase_rel.max() > 1e-2:
        raise ValueError(f"sample mbase disagrees with our pmax by "
                         f"{mbase_rel.max():.3%}; wrong case variant.")
    gen_static[:, 0] = g_sample[:, 0]                         # mbase (MVA)
    gen_static[:, 1] = g_sample[:, 1]                         # pg  init (pu)
    gen_static[:, 2] = gin["min_p_mw"].values / BASE_MVA
    gen_static[:, 3] = gin["max_p_mw"].values / BASE_MVA
    gen_static[:, 4] = g_sample[:, 4]                         # qg  init (pu)
    gen_static[:, 5] = gin["min_q_mvar"].values / BASE_MVA
    gen_static[:, 6] = gin["max_q_mvar"].values / BASE_MVA
    gen_static[:, 7] = 1.0                                    # vg setpoint
    gen_static[:, 8] = gin["cp2_eur_per_mw2"].values * BASE_MVA ** 2
    gen_static[:, 9] = gin["cp1_eur_per_mw"].values * BASE_MVA
    gen_static[:, 10] = gin["cp0_eur"].values

    # --- loads / shunts: fixed node sets, ascending bus order --------------
    load_bus = np.where((bus0["Pd"].values != 0) | (bus0["Qd"].values != 0))[0]
    shunt_bus = np.where((bus0["BS"].values != 0) | (bus0["GS"].values != 0))[0]
    shunt = np.stack([bus0["BS"].values[shunt_bus],
                      bus0["GS"].values[shunt_bus]], axis=1).astype(float)

    # --- branches: in-service only; transformer iff the two ends differ in
    # nominal kV. A tap/shift rule would miss the 88 of 192 transformers with
    # tap == 1, shift == 0.
    br = branch0[branch0["br_status"] > 0]
    kv = bus0["vn_kv"].values
    fb = br["from_bus"].values.astype(int)
    tb = br["to_bus"].values.astype(int)
    is_tr = kv[fb] != kv[tb]

    def _cols(sel: np.ndarray, name: str) -> np.ndarray:
        return br[name].values[sel].astype(float)

    ac = ~is_tr
    ac_rate = _cols(ac, "rate_a") / BASE_MVA
    ac_b_half = _cols(ac, "b") / 2.0
    ac_feats = np.stack([
        np.deg2rad(_cols(ac, "ang_min")), np.deg2rad(_cols(ac, "ang_max")),
        ac_b_half, ac_b_half,
        _cols(ac, "r"), _cols(ac, "x"),
        ac_rate, ac_rate, ac_rate,
    ], axis=1)

    tr_rate = _cols(is_tr, "rate_a") / BASE_MVA
    tr_b_half = _cols(is_tr, "b") / 2.0
    tr_feats = np.stack([
        np.deg2rad(_cols(is_tr, "ang_min")), np.deg2rad(_cols(is_tr, "ang_max")),
        _cols(is_tr, "r"), _cols(is_tr, "x"),
        tr_rate, tr_rate, tr_rate,
        _cols(is_tr, "tap"), np.deg2rad(_cols(is_tr, "shift")),
        tr_b_half, tr_b_half,
    ], axis=1)

    return StaticGrid(
        bus=bus,
        gen_static=gen_static,
        gen_bus=gin["bus"].values.astype(int),
        gen_row_idx=gin["idx"].values,
        gen_is_slack=gin["is_slack_gen"].values.astype(bool),
        load_bus=load_bus,
        shunt_bus=shunt_bus,
        shunt=shunt,
        ac_senders=fb[ac], ac_receivers=tb[ac], ac_feats=ac_feats,
        ac_row_idx=br["idx"].values[ac],
        tr_senders=fb[is_tr], tr_receivers=tb[is_tr], tr_feats=tr_feats,
        tr_row_idx=br["idx"].values[is_tr],
    )


def verify_against_sample(sg: StaticGrid, sample: dict) -> Dict[str, bool]:
    """Assert every field we derived matches the same-case reference."""
    g = sample["grid"]
    checks = {
        "bus": np.allclose(sg.bus, np.asarray(g["nodes"]["bus"], float)),
        "generator": np.allclose(sg.gen_static,
                                 np.asarray(g["nodes"]["generator"], float),
                                 atol=1e-9),
        "shunt": np.allclose(sg.shunt, np.asarray(g["nodes"]["shunt"], float)),
        "ac_senders": np.array_equal(sg.ac_senders,
                                     np.asarray(g["edges"]["ac_line"]["senders"])),
        "ac_receivers": np.array_equal(sg.ac_receivers,
                                       np.asarray(g["edges"]["ac_line"]["receivers"])),
        "ac_features": np.allclose(sg.ac_feats,
                                   np.asarray(g["edges"]["ac_line"]["features"], float)),
        "tr_senders": np.array_equal(sg.tr_senders,
                                     np.asarray(g["edges"]["transformer"]["senders"])),
        "tr_receivers": np.array_equal(sg.tr_receivers,
                                       np.asarray(g["edges"]["transformer"]["receivers"])),
        "tr_features": np.allclose(sg.tr_feats,
                                   np.asarray(g["edges"]["transformer"]["features"], float)),
        "gen_bus_link": np.array_equal(
            sg.gen_bus, np.asarray(g["edges"]["generator_link"]["receivers"])),
        "load_bus_link": np.array_equal(
            sg.load_bus, np.asarray(g["edges"]["load_link"]["receivers"])),
        "shunt_bus_link": np.array_equal(
            sg.shunt_bus, np.asarray(g["edges"]["shunt_link"]["receivers"])),
    }
    return checks


def _links(n_src: int, dst: np.ndarray) -> dict:
    return {"senders": list(range(n_src)), "receivers": [int(b) for b in dst]}


def _flow_labels(branch_s: pd.DataFrame, row_idx: np.ndarray) -> List[List[float]]:
    """Solver branch flows as OPFData `edge_label` rows.

    Column order is [Pji, Qji, Pij, Qij], the TO end first, which is OPFData's
    convention: reconstructing both end injections from an OPFData scenario's
    own label voltages with the pi model matches its
    ``solution.edges.ac_line.features`` in that order to ~4e-12 pu. A flow loss
    on a swapped order would learn sign-inverted targets (Pji ~ -Pij), and
    ``verify_flow_label_order`` checks the first emitted scenario for this.

    Row order must match the edge order, so ``branch_s`` is reindexed by the
    datakit ``idx`` captured when the edges were built.
    """
    sub = branch_s.set_index("idx").loc[row_idx]
    return np.stack([sub["pt"].values / BASE_MVA, sub["qt"].values / BASE_MVA,
                     sub["pf"].values / BASE_MVA, sub["qf"].values / BASE_MVA],
                    axis=1).tolist()


def verify_flow_label_order(obj: dict) -> None:
    """Assert the emitted flow labels follow OPFData's [Pji, Qji, Pij, Qij].

    Recomputes both branch-end injections from the scenario's own solution
    voltages with the pi model and checks which columns they land in.
    """
    g, sol = obj["grid"], obj["solution"]
    bus = np.asarray(sol["nodes"]["bus"], dtype=float)
    v = bus[:, 1] * np.exp(1j * bus[:, 0])
    feats = np.asarray(g["edges"]["ac_line"]["features"], dtype=float)
    snd = np.asarray(g["edges"]["ac_line"]["senders"], dtype=int)
    rcv = np.asarray(g["edges"]["ac_line"]["receivers"], dtype=int)
    lab = np.asarray(sol["edges"]["ac_line"]["features"], dtype=float)

    y = 1.0 / (feats[:, 4] + 1j * feats[:, 5])
    vi, vj = v[snd], v[rcv]
    s_ij = vi * np.conj(y * (vi - vj) + 1j * feats[:, 2] * vi)   # FROM end
    s_ji = vj * np.conj(y * (vj - vi) + 1j * feats[:, 3] * vj)   # TO end

    tol = 1e-6
    checks = {
        "col0 == Pji": np.abs(lab[:, 0] - s_ji.real).max(),
        "col1 == Qji": np.abs(lab[:, 1] - s_ji.imag).max(),
        "col2 == Pij": np.abs(lab[:, 2] - s_ij.real).max(),
        "col3 == Qij": np.abs(lab[:, 3] - s_ij.imag).max(),
    }
    bad = {k: v for k, v in checks.items() if v > tol}
    if bad:
        raise SystemExit(
            "flow label column order is wrong (OPFData wants [Pji,Qji,Pij,Qij]); "
            f"max residuals: {bad}"
        )
    print("  flow label order [Pji,Qji,Pij,Qij] verified "
          f"(max residual {max(checks.values()):.2e} pu)")


def build_scenario(sg: StaticGrid, bus_s: pd.DataFrame, gen_s: pd.DataFrame,
                   branch_s: pd.DataFrame, sid: int) -> dict:
    """Assemble one `.pyg.json` object for a single scenario."""
    load = np.stack([
        bus_s["Pd"].values[sg.load_bus] / BASE_MVA,
        bus_s["Qd"].values[sg.load_bus] / BASE_MVA,
    ], axis=1)

    gin = gen_s[gen_s["in_service"] > 0]
    pg = gin["p_mw"].values / BASE_MVA
    qg = gin["q_mvar"].values / BASE_MVA

    # quadratic cost over all in-service units, MW basis
    p_mw = gin["p_mw"].values
    objective = float((gin["cp2_eur_per_mw2"].values * p_mw ** 2
                       + gin["cp1_eur_per_mw"].values * p_mw
                       + gin["cp0_eur"].values).sum())

    sol_bus = np.stack([np.deg2rad(bus_s["Va"].values),
                        bus_s["Vm"].values], axis=1)

    return {
        "grid": {
            "nodes": {
                "bus": sg.bus.tolist(),
                "generator": sg.gen_static.tolist(),
                "load": load.tolist(),
                "shunt": sg.shunt.tolist(),
            },
            "edges": {
                "ac_line": {
                    "senders": sg.ac_senders.tolist(),
                    "receivers": sg.ac_receivers.tolist(),
                    "features": sg.ac_feats.tolist(),
                },
                "transformer": {
                    "senders": sg.tr_senders.tolist(),
                    "receivers": sg.tr_receivers.tolist(),
                    "features": sg.tr_feats.tolist(),
                },
                "generator_link": _links(len(sg.gen_bus), sg.gen_bus),
                "load_link": _links(len(sg.load_bus), sg.load_bus),
                "shunt_link": _links(len(sg.shunt_bus), sg.shunt_bus),
            },
            "context": [[[BASE_MVA]]],
        },
        "solution": {
            "nodes": {
                "bus": sol_bus.tolist(),
                "generator": np.stack([pg, qg], axis=1).tolist(),
            },
            # Branch flow labels feed the flow-regression and thermal terms of
            # a full-state training loss.
            "edges": {
                "ac_line": {
                    "senders": sg.ac_senders.tolist(),
                    "receivers": sg.ac_receivers.tolist(),
                    "features": _flow_labels(branch_s, sg.ac_row_idx),
                },
                "transformer": {
                    "senders": sg.tr_senders.tolist(),
                    "receivers": sg.tr_receivers.tolist(),
                    "features": _flow_labels(branch_s, sg.tr_row_idx),
                },
            },
        },
        "metadata": {
            "objective": objective,
            "termination_status": "LOCALLY_SOLVED",
            "feasible": True,
            "scenario_id": int(sid),
            "source_variant": "gridfm_datakit_case500_goc",
            "bus_id_map": list(range(1, len(sg.bus) + 1)),
            "gen_id_map": [int(i) + 1 for i in sg.gen_row_idx],
            "gen_bus_map": [int(b) + 1 for b in sg.gen_bus],
            "load_bus_map": [int(b) + 1 for b in sg.load_bus],
            "gen_is_slack": [bool(x) for x in sg.gen_is_slack],
            "export_schema_version": 2,
            "flow_label_order": "Pji,Qji,Pij,Qij",
        },
    }


def all_scenario_ids(raw_dir: Path) -> np.ndarray:
    sids = pd.read_parquet(raw_dir / "bus_data.parquet",
                           columns=["load_scenario_idx"])["load_scenario_idx"]
    return np.sort(sids.unique())


def split_scenario_ids(raw_dir: Path, split: str, limit: int = 0) -> np.ndarray:
    """Scenario ids for `split`. The two splits are disjoint by construction."""
    uniq = all_scenario_ids(raw_dir)
    if split == "holdout":
        ids = uniq[-N_HOLDOUT:]
        if ids[0] != HOLDOUT_START:
            raise ValueError(f"holdout starts at {ids[0]}, expected {HOLDOUT_START}")
    else:
        ids = uniq[uniq < HOLDOUT_START]
        if limit:
            # Evenly spaced subsample so a capped run still spans the whole
            # load-factor range instead of only the earliest scenarios.
            ids = ids[np.linspace(0, len(ids) - 1, int(limit)).round().astype(int)]
    return ids


def _read_chunk(sids: np.ndarray, raw_dir: Path = RAW_DIR) -> tuple:
    flt = [("load_scenario_idx", "in", [float(s) for s in sids])]
    return (pd.read_parquet(raw_dir / "bus_data.parquet", filters=flt),
            pd.read_parquet(raw_dir / "gen_data.parquet", filters=flt),
            pd.read_parquet(raw_dir / "branch_data.parquet", filters=flt))


def static_grid_for(ids: np.ndarray, raw_dir: Path, sample: dict) -> StaticGrid:
    """Build the scenario-invariant static grid from the first scenario and
    verify it."""
    bus0_df, gen0_df, br0_df = _read_chunk(ids[:1], raw_dir)
    sg = build_static_grid(
        bus0_df.sort_values("bus").reset_index(drop=True),
        gen0_df.sort_values("idx").reset_index(drop=True),
        br0_df.sort_values("idx").reset_index(drop=True),
        sample,
    )
    checks = verify_against_sample(sg, sample)
    print("structural verification vs the same-case OPFData reference:")
    for k, ok in checks.items():
        print(f"  {k:16s} {'OK' if ok else 'MISMATCH'}")
    if not all(checks.values()):
        raise SystemExit("static grid does not match the same-case reference; aborting.")
    print(f"  {len(sg.bus)} buses, {len(sg.gen_bus)} in-service gens "
          f"({int(sg.gen_is_slack.sum())} slack), {len(sg.load_bus)} loads, "
          f"{len(sg.shunt_bus)} shunts, {len(sg.ac_feats)} ac_lines, "
          f"{len(sg.tr_feats)} transformers")
    return sg


def iter_scenarios(sg: StaticGrid, ids: np.ndarray, raw_dir: Path = RAW_DIR,
                   chunk: int = 500):
    """Yield ``(sid, scenario_dict)`` in ``ids`` order, ``chunk`` at a time."""
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        bus, gen, branch = _read_chunk(part, raw_dir)
        bus_g = dict(tuple(bus.groupby("load_scenario_idx")))
        gen_g = dict(tuple(gen.groupby("load_scenario_idx")))
        br_g = dict(tuple(branch.groupby("load_scenario_idx")))
        for sid in part:
            yield sid, build_scenario(sg, bus_g[sid].sort_values("bus"),
                                      gen_g[sid].sort_values("idx"), br_g[sid],
                                      int(sid))


def scenario_filename(sid) -> str:
    return f"scenario_{int(sid):05d}.pyg.json"


def scenario_ids_on_disk(out_dir: Path) -> list[int]:
    return sorted(int(p.name[len("scenario_"):-len(".pyg.json")])
                  for p in Path(out_dir).glob("scenario_*.pyg.json"))


def converted_digest(out_dir: Path) -> dict:
    """SHA-256 over every converted file, in scenario-id order.

    Each file contributes ``<name> <sha256 of its bytes>\\n`` to an outer
    digest, so the result pins the set of scenarios and every byte of each.
    """
    outer = hashlib.sha256()
    n_bytes = 0
    ids = scenario_ids_on_disk(out_dir)
    for sid in ids:
        data = (Path(out_dir) / scenario_filename(sid)).read_bytes()
        n_bytes += len(data)
        outer.update(f"{scenario_filename(sid)} "
                     f"{hashlib.sha256(data).hexdigest()}\n".encode())
    return {"files": len(ids), "first": ids[0] if ids else None,
            "last": ids[-1] if ids else None, "bytes": n_bytes,
            "sha256": outer.hexdigest()}


def raw_summary(raw_dir: Path = RAW_DIR) -> dict:
    """What a datakit run of ``config_case500_30k.yaml`` leaves in ``raw/``.

    The scenario count datakit reports, the solved ids, the generated ids
    missing from the solved tables (OPF did not converge), and a digest of the
    sampled load profile. The profile is drawn before any of the 30,000 solves,
    so a regeneration can be checked against it after minutes rather than after
    the ~12 hours the solves take.
    """
    raw_dir = Path(raw_dir)
    ids = all_scenario_ids(raw_dir).astype(np.int64)
    t = pd.read_parquet(raw_dir / "scenarios_agg_load_profile.parquet",
                        columns=["load_scenario", "load", "p_mw", "q_mvar"])
    h = hashlib.sha256()
    for col in ("load_scenario", "load", "p_mw", "q_mvar"):
        h.update(np.ascontiguousarray(t[col].to_numpy()).tobytes())
    n_generated = int(t["load_scenario"].max()) + 1
    return {
        "n_scenarios_txt": int((raw_dir / "n_scenarios.txt").read_text().strip()),
        "n_generated": n_generated,
        "n_solved": int(ids.size),
        "unsolved_ids": sorted(set(range(n_generated)) - set(ids.tolist())),
        "load_profile_rows": int(len(t)),
        "load_profile_sha256": h.hexdigest(),
    }


def verify_outputs(out_dir: Path, split: str, deep: bool = False,
                   expected: dict | None = None) -> list[str]:
    """Problems with a converted split on disk; empty means OK.

    Shallow: the scenario ids match the split definition (holdout =
    29500..29999 exactly; train = 29,487 ids, all below 29500). Deep: the content
    digest equals the published conversion's (``checksums.json``). IPOPT solves
    are not guaranteed bit-reproducible across machines, so a deep mismatch on a
    regenerated corpus means "not the published bytes", not "broken".
    """
    ids = scenario_ids_on_disk(out_dir)
    problems = []
    if split == "holdout":
        if ids != list(range(HOLDOUT_START, HOLDOUT_START + N_HOLDOUT)):
            problems.append(f"holdout has {len(ids)} files, expected ids "
                            f"{HOLDOUT_START}..{HOLDOUT_START + N_HOLDOUT - 1}")
    else:
        if len(ids) != N_TRAIN:
            problems.append(f"train has {len(ids)} files, expected {N_TRAIN}")
        if ids and ids[-1] >= HOLDOUT_START:
            problems.append(f"train contains holdout ids (max {ids[-1]})")
    if deep and not problems:
        got = converted_digest(out_dir)
        ref = (expected or {}).get(f"case500_datakit/{split}_opfdata")
        if ref is not None and ref.get("sha256") != got["sha256"]:
            problems.append(f"content sha256 {got['sha256'][:16]}... != "
                            f"published {ref['sha256'][:16]}...")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["holdout", "train"], default="holdout")
    ap.add_argument("--limit", type=int, default=0,
                    help="train split only: evenly spaced subsample of N scenarios")
    ap.add_argument("--chunk", type=int, default=500,
                    help="scenarios held in memory at once")
    ap.add_argument("--raw", default=str(RAW_DIR),
                    help="datakit raw/ directory (default: under the data home)")
    ap.add_argument("--reference", default=str(REFERENCE_PATH),
                    help="same-case OPFData scenario: a consolidated shard "
                         "(first line), a .json.gz, or a .json/.pyg.json")
    ap.add_argument("--out", default=None,
                    help="output directory (default: case500_datakit/"
                         "{holdout,train}_opfdata under the data home)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing conversion instead of writing one")
    ap.add_argument("--deep", action="store_true",
                    help="with --verify-only: also compare the content digest "
                         "with the published conversion")
    args = ap.parse_args()

    out_dir = (Path(args.out).resolve() if args.out
               else OUT_DIR if args.split == "holdout" else TRAIN_OUT_DIR)
    if args.verify_only:
        expected = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
        problems = verify_outputs(out_dir, args.split, args.deep, expected)
        print(f"[{'OK' if not problems else 'FAIL'}] {args.split} ({out_dir})")
        for p in problems:
            print(f"       {p}")
        raise SystemExit(1 if problems else 0)

    raw_dir = Path(args.raw).resolve()
    require_raw(raw_dir)
    sample = read_reference(Path(args.reference).resolve())
    ids = split_scenario_ids(raw_dir, args.split, args.limit)
    print(f"split={args.split}: {len(ids)} scenarios, "
          f"load_scenario_idx {ids[0]:.0f}..{ids[-1]:.0f} -> {out_dir}")

    if args.split == "train" and (ids >= HOLDOUT_START).any():
        raise SystemExit("train split overlaps the holdout; aborting.")

    sg = static_grid_for(ids, raw_dir, sample)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    checked_flow_order = False
    for sid, obj in iter_scenarios(sg, ids, raw_dir, args.chunk):
        if not checked_flow_order:
            verify_flow_label_order(obj)
            checked_flow_order = True
        (out_dir / scenario_filename(sid)).write_text(json.dumps(obj))
        n_written += 1
        if n_written % args.chunk == 0 or n_written == len(ids):
            print(f"  {n_written}/{len(ids)}", flush=True)
    print(f"wrote {n_written} files to {out_dir}")


if __name__ == "__main__":
    main()
