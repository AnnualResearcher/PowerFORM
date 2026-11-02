"""Figure 1 of the paper, drawn from the reconfiguration run records.

Reconfiguration transfer for both map families, 1 x 4 at the text width:

  (a) physics maps       median loss gap to the greedy reference
  (b) generic attention  median loss gap
  (c) physics maps       infeasible fraction
  (d) generic attention  infeasible fraction

Each metric is a pair sharing its y axis. Every complete seed pair of a family is
drawn. Bands are min-max over seeds, the line is the seed mean, and nothing is
smoothed. Means above the gap panels' y clip are drawn as edge triangles. Dashed
lines: zero gap, the greedy reference itself, and the rate at which none of a
scenario's random three-line openings was feasible.

    python -m powerform.experiments.figure [--results DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from powerform.paths import OUTPUTS, RESULTS  # noqa: E402

OUT_DIR = OUTPUTS / "figures"
NAME = "figure1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ARMS = {
    "pretrained": {"label": "pretrained", "color": "#1f4e79"},
    "scratch": {"label": "scratch", "color": "#c0504d"},
}
# The early transient of the gap reaches a few percent, while the result is the
# approach to and below zero.
A_YMAX, A_YMIN = 1.5, -0.65
FAMILIES = (("physics", "physics maps"), ("generic", "generic attention"))
X_MAX = 20000
LW = 1.0
TOP = 0.86        # axes top as a figure fraction
# No creation timestamp in the PDF, so two runs on the same inputs write the same
# bytes. The PNG writer stores no timestamp to begin with.
PDF_METADATA = {"CreationDate": None}


def random_infeasible_rate(results: Path) -> float:
    """Fraction of scored scenarios where none of the random openings was feasible."""
    path = results / "reconfiguration" / "reference_val.json"
    if not path.exists():
        raise SystemExit(f"no reference table at {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    usable = [r for r in rows if r["ref_pu"] is not None]
    return sum(1 for r in usable if not r["n_random_converged"]) / len(usable)


def complete_seeds(results: Path, maps: str) -> list[str]:
    """Seeds with a finished record for both arms, so the two bands have the same
    sample count."""
    per_arm = [{f.name.rsplit("seed", 1)[1].split(".")[0]
                for f in (results / "reconfiguration").glob(
                    f"{maps}_{arm}_seed*.json")}
               for arm in ARMS]
    seeds = sorted(set.intersection(*per_arm), key=int)
    if not seeds:
        raise SystemExit(f"no complete seed pair for {maps} in {results}")
    return seeds


def load(results: Path, maps: str, arm: str, seeds: list[str]) -> tuple:
    """{step: [value per seed]} for gap and infeasible fraction. Steps that not
    every seed reached are dropped rather than averaged over fewer runs."""
    gap, inf = defaultdict(list), defaultdict(list)
    for s in seeds:
        f = results / "reconfiguration" / f"{maps}_{arm}_seed{s}.json"
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if r["metric"] == "gap_pct_median":
                gap[int(r["step"])].append(float(r["value"]))
            elif r["metric"] == "infeasible_fraction":
                inf[int(r["step"])].append(float(r["value"]))
    full = lambda d: {t: v for t, v in d.items() if len(v) == len(seeds)}  # noqa: E731
    return full(gap), full(inf)


def draw(ax, per_step: dict, style: dict, label, clip: tuple | None) -> None:
    """Min-max band and seed-mean line; with `clip`, means above the top are
    replaced by edge triangles instead of being drawn off the axes."""
    steps = np.array(sorted(per_step))
    vals = [per_step[t] for t in steps]
    mean = np.array([np.mean(v) for v in vals])
    lo = np.array([np.min(v) for v in vals])
    hi = np.array([np.max(v) for v in vals])
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(steps, lo, hi, color=style["color"], alpha=0.16, linewidth=0)
    if clip is not None:
        off = mean > clip[1]
        if off.any():
            ax.plot(steps[off], np.full(off.sum(), clip[1] * 0.985), "^",
                    color=style["color"], markersize=2.6, markeredgewidth=0,
                    clip_on=False, zorder=4)
        mean = np.where(off, np.nan, mean)
    ax.plot(steps, mean, color=style["color"], linewidth=LW, label=label, zorder=3)


def style_axes(ax, ref: float, ylim: tuple, first_of_pair: bool) -> None:
    ax.axhline(ref, color="0.45", linestyle=(0, (4, 2)), linewidth=0.6, zorder=1)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    # The right panel of a pair drops its "0" so it cannot run into the left
    # panel's "20k".
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: (f"{x / 1000:.0f}k" if x else ("0" if first_of_pair else ""))))
    ax.set_xlabel("training steps")
    ax.grid(True, linewidth=0.3, color="0.90", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not first_of_pair:
        ax.tick_params(labelleft=False)


def build(results: Path = RESULTS):
    """Draw the four panels from the records in `results`. The figure is left open
    so a caller can inspect what was plotted before saving or closing it."""
    results = Path(results)
    fig = plt.figure(figsize=(5.5, 1.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.30, 1, 1], wspace=0.10,
                          left=0.075, right=0.98, top=TOP, bottom=0.36)
    ax = [fig.add_subplot(gs[0, j]) for j in (0, 1, 3, 4)]
    rand_rate = random_infeasible_rate(results)
    for fi, (maps, title) in enumerate(FAMILIES):
        seeds = complete_seeds(results, maps)
        for arm, st in ARMS.items():
            gap, inf = load(results, maps, arm, seeds)
            draw(ax[fi], gap, st, st["label"] if fi == 0 else None, clip=(A_YMIN, A_YMAX))
            draw(ax[2 + fi], inf, st, None, clip=None)
        style_axes(ax[fi], 0.0, (A_YMIN, A_YMAX), first_of_pair=(fi == 0))
        style_axes(ax[2 + fi], rand_rate, (-0.03, 1.03), first_of_pair=(fi == 0))
        for j, letters in ((fi, "ab"), (2 + fi, "cd")):
            ax[j].set_title(f"({letters[fi]}) PowerFORM\n{title}", loc="left",
                            fontsize=8, pad=2, linespacing=1.05)
    ax[0].set_ylabel("eval gap (%)")
    ax[2].set_ylabel("infeasible fraction")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.53, 0.0),
               handlelength=1.6, columnspacing=1.4)
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results tree holding reconfiguration/")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig = build(a.results)
    fig.savefig(a.out_dir / f"{NAME}.pdf", metadata=dict(PDF_METADATA))
    fig.savefig(a.out_dir / f"{NAME}.png", dpi=400)
    plt.close(fig)
    print(f"-> {a.out_dir / NAME}.pdf, .png")


if __name__ == "__main__":
    main()
