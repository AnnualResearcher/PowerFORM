"""Paired step-window comparisons and first crossings of the reconfiguration runs.

Two reductions per map family, over every seed with both arms:

  windows    Per (seed, window), each arm's value is the mean over the eval steps
             inside the window, and diff = pretrained - scratch (negative =
             pretrained better). Per window: the seed mean of diff, the per-seed
             diffs, `wins` (seeds with a negative diff) and each arm's seed mean.
             A seed missing a step of a window is left out of that window.
  crossings  The first step at which the seed-mean curve, smoothed by a trailing
             3-point running mean, reaches an infeasible fraction <= 0.5 or a
             median gap <= 0; null when it never does. Also the range of each arm's
             seed-mean infeasible fraction over 5k-20k steps.

Window means use statistics.mean, which rounds exactly, and the crossings use
np.mean. The two differ in the last bit, so neither may replace the other without
changing the recorded numbers.

    python -m powerform.experiments.windows [--maps physics] [--results DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

from powerform.paths import OUTPUTS, RESULTS

ARMS = ("pretrained", "scratch")
WIN = {"50-250": range(50, 251, 50), "300-500": range(300, 501, 50),
       "550-1000": range(550, 1001, 50), "1250-2000": range(1250, 2001, 250),
       "2250-5000": range(2250, 5001, 250), "16k-20k": range(16000, 20001, 1000)}
METRICS = ("infeasible_fraction", "gap_pct", "gap_pct_median")
THRESHOLDS = (("infeasible_fraction", 0.5), ("gap_pct_median", 0.0))
LATE = (5000, 20000)
OUT_DIR = OUTPUTS / "reconfiguration"


def run_path(results: Path, maps: str, arm: str, seed) -> Path:
    return results / "reconfiguration" / f"{maps}_{arm}_seed{seed}.json"


def complete_pairs(results: Path, maps: str) -> list[int]:
    """Every seed with both arms present."""
    seeds = []
    for f in (results / "reconfiguration").glob(f"{maps}_pretrained_seed*.json"):
        seed = re.search(r"seed(\d+)\.json$", f.name).group(1)
        if run_path(results, maps, "scratch", seed).exists():
            seeds.append(int(seed))
    return sorted(seeds)


def series(results: Path, maps: str, arm: str, seed, metric: str) -> dict:
    d = json.loads(run_path(results, maps, arm, seed).read_text(encoding="utf-8"))
    return {r["step"]: r["value"] for r in d["rows"] if r["metric"] == metric}


def paired_windows(results: Path, maps: str, seeds: list[int]) -> dict:
    """The windows reduction, as the record that is written."""
    metrics = {}
    for metric in METRICS:
        diffs = {w: [] for w in WIN}
        pre = {w: [] for w in WIN}
        scr = {w: [] for w in WIN}
        for s in seeds:
            sw = series(results, maps, "pretrained", s, metric)
            ss = series(results, maps, "scratch", s, metric)
            for w, steps in WIN.items():
                pts = [t for t in steps if t in sw and t in ss]
                if len(pts) != len(steps):
                    continue
                dw = st.mean(sw[t] for t in pts)
                ds = st.mean(ss[t] for t in pts)
                diffs[w].append(dw - ds)
                pre[w].append(dw)
                scr[w].append(ds)
        metrics[metric] = {
            w: {"diff_mean": st.mean(diffs[w]) if diffs[w] else None,
                "diff_per_seed": diffs[w],
                "wins": sum(x < 0 for x in diffs[w]), "n": len(diffs[w]),
                "pretrained_abs": st.mean(pre[w]) if pre[w] else None,
                "scratch_abs": st.mean(scr[w]) if scr[w] else None}
            for w in WIN}
    return {"maps": maps, "seeds": seeds,
            "windows": {k: [min(v), max(v)] for k, v in WIN.items()},
            "metrics": metrics}


def seed_mean_curve(results: Path, maps: str, arm: str, seeds: list[int],
                    metric: str) -> tuple[list[int], np.ndarray]:
    """Steps every seed reached, and the seed-mean value at each."""
    per = [{int(t): v for t, v in series(results, maps, arm, s, metric).items()}
           for s in seeds]
    steps = sorted(set.intersection(*[set(p) for p in per]))
    if not steps:
        raise SystemExit(f"{maps} {arm}: no {metric} step shared by seeds {seeds}")
    return steps, np.array([np.mean([p[t] for p in per]) for t in steps])


def first_crossing(steps: list[int], mean: np.ndarray, thr: float):
    """First step where the trailing 3-point running mean is <= thr, or None."""
    sm = np.array([mean[max(0, i - 2):i + 1].mean() for i in range(len(mean))])
    return next((steps[i] for i in range(len(sm)) if sm[i] <= thr), None)


def crossings(results: Path, maps: str, seeds: list[int]) -> dict:
    """The crossings reduction, as the record that is written."""
    first = {}
    for metric, thr in THRESHOLDS:
        first[metric] = {"threshold": thr}
        for arm in ARMS:
            steps, mean = seed_mean_curve(results, maps, arm, seeds, metric)
            first[metric][arm] = first_crossing(steps, mean, thr)
    late = {}
    for arm in ARMS:
        steps, mean = seed_mean_curve(results, maps, arm, seeds,
                                      "infeasible_fraction")
        vals = [float(v) for t, v in zip(steps, mean) if LATE[0] <= t <= LATE[1]]
        if not vals:
            raise SystemExit(f"{maps}: no eval step in {LATE[0]}-{LATE[1]} reached "
                             "by every seed")
        late[arm] = {"min": min(vals), "max": max(vals), "n_points": len(vals)}
    return {"maps": maps, "seeds": seeds, "first_crossing": first,
            "late_seed_mean_infeasible_fraction": {"steps": list(LATE), **late}}


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
        cross = crossings(a.results, maps, seeds)
        windows = paired_windows(a.results, maps, seeds)
        (a.out_dir / f"windows_{maps}.json").write_text(json.dumps(windows, indent=1))
        (a.out_dir / f"crossings_{maps}.json").write_text(json.dumps(cross, indent=1))
        for metric, thr in THRESHOLDS:
            c = cross["first_crossing"][metric]
            print(f"{maps} seeds {seeds} | {metric} <= {thr}: pretrained "
                  f"{c['pretrained']} | scratch {c['scratch']}")
    print(f"-> {a.out_dir}")


if __name__ == "__main__":
    main()
