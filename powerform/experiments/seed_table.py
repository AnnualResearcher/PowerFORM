"""Per-seed table of the reconfiguration runs.

For each seed with both arms: the mean of each metric over the first 1,000 and
3,000 steps with the number of eval steps each arm wins, the infeasible fraction
from 5,000 steps on, and the terminal gap. An arm counts as ahead only when its
[min, max] band over at least three seeds does not overlap the other arm's.

    python -m powerform.experiments.seed_table [--maps physics] [--results DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from powerform.experiments.windows import OUT_DIR, complete_pairs, run_path
from powerform.paths import RESULTS

AXES = [("infeasible_fraction", "infeasible"),
        ("gap_pct_median", "gap median"),
        ("gap_pct", "gap mean")]
MIN_SEEDS = 3


def curve(results: Path, maps: str, arm: str, seed: int) -> dict[int, dict]:
    d = json.loads(run_path(results, maps, arm, seed).read_text(encoding="utf-8"))
    by: dict = collections.defaultdict(dict)
    for r in d["rows"]:
        by[int(r["step"])][r["metric"]] = r["value"]
    return by


def band(vals: list[float]) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.min()), float(a.max())


def ahead(pre: list[float], scr: list[float]) -> str:
    """Which arm's band lies entirely below the other's."""
    if len(pre) < MIN_SEEDS:
        return f"too few seeds (n={len(pre)})"
    _, plo, phi = band(pre)
    _, slo, shi = band(scr)
    if phi < slo:
        return "pretrained lower"
    if plo > shi:
        return "scratch lower"
    return "bands overlap"


def seed_row(results: Path, maps: str, seed: int) -> dict:
    """The table's numbers for one seed pair."""
    P, S = curve(results, maps, "pretrained", seed), curve(results, maps, "scratch", seed)
    common = sorted(set(P) & set(S))
    row: dict = {"n_grid_points": len(common)}
    for lim in (1000, 3000):
        pts = [x for x in common if x <= lim]
        for key, name in AXES:
            pv = [P[x][key] for x in pts if key in P[x]]
            sv = [S[x][key] for x in pts if key in S[x]]
            if not pv or len(pv) != len(sv):
                continue
            row[f"<={lim}|{name}|pretrained"] = float(np.mean(pv))
            row[f"<={lim}|{name}|scratch"] = float(np.mean(sv))
            row[f"<={lim}|{name}|wins"] = int(sum(1 for x, y in zip(pv, sv) if x < y))
            row[f"<={lim}|{name}|n"] = len(pv)
    late = [x for x in common if x >= 5000]
    row[">=5000|infeasible|pretrained"] = float(
        np.mean([P[x]["infeasible_fraction"] for x in late]))
    row[">=5000|infeasible|scratch"] = float(
        np.mean([S[x]["infeasible_fraction"] for x in late]))
    t = max(common)
    row["terminal|step"] = t
    row["terminal|gap|pretrained"] = P[t]["gap_pct"]
    row["terminal|gap|scratch"] = S[t]["gap_pct"]
    return row


def report(per_seed: dict, seeds: list[int]) -> list[str]:
    """Bands over the seeds, for the terminal."""
    prefixes = [f"<={lim}|{name}" for lim in (1000, 3000) for _, name in AXES
                if f"<={lim}|{name}|pretrained" in per_seed[seeds[0]]]
    prefixes += [">=5000|infeasible", "terminal|gap"]
    lines = []
    for prefix in prefixes:
        pre = [per_seed[s][f"{prefix}|pretrained"] for s in seeds]
        scr = [per_seed[s][f"{prefix}|scratch"] for s in seeds]
        pm, plo, phi = band(pre)
        sm, slo, shi = band(scr)
        lines.append(f"  {prefix:<22} pretrained {pm:7.3f} [{plo:.3f}, {phi:.3f}] | "
                     f"scratch {sm:7.3f} [{slo:.3f}, {shi:.3f}]  {ahead(pre, scr)}")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--maps", choices=["physics", "generic"], default=None,
                    help="one map family; default both")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results tree holding reconfiguration/<maps>_<arm>_seed*.json")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for maps in ([a.maps] if a.maps else ["physics", "generic"]):
        seeds = complete_pairs(a.results, maps)
        if not seeds:
            raise SystemExit(f"no complete seed pair for {maps} in {a.results}")
        per_seed = {s: seed_row(a.results, maps, s) for s in seeds}
        print(f"{maps} pairs, seeds {seeds}")
        print("\n".join(report(per_seed, seeds)))
        out = a.out_dir / f"seed_table_{maps}.json"
        out.write_text(json.dumps(
            {"maps": maps, "seeds": seeds,
             "per_seed": {str(k): v for k, v in per_seed.items()}}, indent=1))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
