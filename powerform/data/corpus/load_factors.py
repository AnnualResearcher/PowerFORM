"""Measure the operating band of the in-distribution corpus (App. B).

The nominal demand is a proxy, because a pglib case file has one load vector: the
median, over 400 OPFData ``case500_goc`` fulltop train scenarios, of the
scenario's total active demand in MW. OPFData perturbs each load independently,
so a scenario's total averages out close to the case-file value.

Factors are each datakit scenario's total active demand over that nominal, read
from three places that must agree on the shared scenarios:

  * ``profile``   datakit's sampled load profile, all 30,000 scenarios;
  * ``solved``    the solved corpus (``bus_data.parquet`` ``Pd``), 29,987;
  * ``converted`` the converted hold-out files, 500.

The generator's own band, ``l`` and ``u`` in its log, scales the aggregate
profile; the per-load noise moves a scenario's realised total slightly outside it.

Usage (from the repository root):
    python -m powerform.data.corpus.load_factors
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from powerform.data.corpus.datakit_export import (
    BASE_MVA, HOLDOUT_START, OUT_DIR, RAW_DIR, require_raw, scenario_filename,
    scenario_ids_on_disk,
)
from powerform.data.opfdata import DATA_ROOT, iter_raw, read_manifest, subsample
from powerform.data.opfdata_case import L_PD, _scatter_sum

N_NOMINAL = 400


def opfdata_totals(case: str = "case500", variant: str = "fulltop",
                   split: str = "train", n: int = N_NOMINAL,
                   root: Path = DATA_ROOT) -> np.ndarray:
    """Total active demand (pu) of the ``n`` scenarios ``load_pool`` would pick.
    Streams the shard, one scenario dict in memory at a time."""
    man = read_manifest(case, variant, root)
    wanted = set(subsample(sorted(man["splits"][split]["example_indices"]), n))
    totals = []
    for _idx, obj in iter_raw(case, variant, split, root, wanted=wanted):
        n_bus = len(obj["grid"]["nodes"]["bus"])
        load = np.asarray(obj["grid"]["nodes"]["load"], dtype=float).reshape(-1, 2)
        load_bus = np.asarray(obj["grid"]["edges"]["load_link"]["receivers"],
                              dtype=np.int64)
        totals.append(float(_scatter_sum(load[:, L_PD], load_bus, n_bus).sum()))
    if len(totals) != len(wanted):
        raise SystemExit(f"read {len(totals)} of {len(wanted)} wanted scenarios")
    return np.asarray(totals)


def profile_totals(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray]:
    """``(scenario_ids, total MW)`` from datakit's sampled load profile."""
    t = pq.read_table(raw_dir / "scenarios_agg_load_profile.parquet",
                      columns=["load_scenario", "p_mw"])
    sid = t.column("load_scenario").to_numpy()
    tot = np.bincount(sid, weights=t.column("p_mw").to_numpy())
    ids = np.unique(sid)
    return ids, tot[ids]


def solved_totals(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray]:
    """``(scenario_ids, total MW)`` from the solved corpus' bus table."""
    ds = pads.dataset(raw_dir / "bus_data.parquet", format="parquet",
                      partitioning="hive")
    t = ds.to_table(columns=["load_scenario_idx", "Pd"])
    sid = t.column("load_scenario_idx").to_numpy().astype(np.int64)
    tot = np.bincount(sid, weights=t.column("Pd").to_numpy())
    ids = np.unique(sid)
    return ids, tot[ids]


def converted_totals(out_dir: Path = OUT_DIR) -> tuple[np.ndarray, np.ndarray]:
    """``(scenario_ids, total MW)`` from converted scenario files."""
    ids = scenario_ids_on_disk(out_dir)
    tot = []
    for sid in ids:
        obj = json.loads((Path(out_dir) / scenario_filename(sid)).read_text())
        tot.append(float(np.asarray(obj["grid"]["nodes"]["load"],
                                    dtype=float)[:, 0].sum()) * BASE_MVA)
    return np.asarray(ids), np.asarray(tot)


def generator_band(raw_dir: Path = RAW_DIR) -> dict:
    """datakit's ``l``/``u`` scaling bounds, as its generation log records them."""
    text = (raw_dir / "scenarios_agg_load_profile.log").read_text()
    out = {}
    for key in ("l", "u"):
        hits = re.findall(rf"^{key}=([0-9.eE+-]+)\s*$", text, flags=re.M)
        if hits:
            out[key] = float(hits[-1])
    return out


def band(ids: np.ndarray, tot: np.ndarray, nominal: float) -> dict:
    """Factor range overall, on the hold-out (ids >= 29500) and on train."""
    f = tot / nominal
    out = {}
    for name, m in (("corpus", ids >= 0), ("holdout", ids >= HOLDOUT_START),
                    ("train", ids < HOLDOUT_START)):
        if m.any():
            out[name] = {"n": int(m.sum()), "min": float(f[m].min()),
                         "max": float(f[m].max())}
    return out


def measure(raw_dir: Path = RAW_DIR, holdout_dir: Path = OUT_DIR,
            root: Path = DATA_ROOT, n_nominal: int = N_NOMINAL) -> dict:
    opf = opfdata_totals(n=n_nominal, root=root)
    nominal = float(np.median(opf)) * BASE_MVA
    res = {"nominal_mw": nominal, "n_nominal": int(opf.size),
           "generator_band": generator_band(raw_dir)}
    for name, fn, arg in (("profile", profile_totals, raw_dir),
                          ("solved", solved_totals, raw_dir),
                          ("converted", converted_totals, holdout_dir)):
        ids, tot = fn(arg)
        res[name] = band(ids, tot, nominal)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--holdout", default=str(OUT_DIR))
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()
    require_raw(Path(args.raw).resolve())
    res = measure(Path(args.raw).resolve(), Path(args.holdout).resolve())
    print(f"nominal = median OPFData case500 train total over "
          f"{res['n_nominal']} scenarios = {res['nominal_mw']:.1f} MW")
    print(f"datakit generator band: {res['generator_band']}")
    for src in ("profile", "solved", "converted"):
        for split, b in res[src].items():
            print(f"  {src:9s} {split:8s} n={b['n']:6d}  "
                  f"factor {b['min']:.3f} .. {b['max']:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
