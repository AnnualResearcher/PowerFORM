"""Figure 2 of the paper (App. D): cost MAPE on the Table 1 hold-out against training samples.

Lines are the 5k-step milestones of the 120k-step (480k-sample) seed-0 runs, taken while
their cosine schedule is still above the floor, so they are not what a shorter budget
reaches. Hollow markers are what it reaches: the ends of Table 1's 20k-step runs (three
seeds) and of the 60k-step seed-0 runs, whose schedules end there. The lower panel is the
schedule the lines ran under. The x-axis counts samples (steps x batch 4), as Table 1 does.

The long-run records come from `results/long_run/`, or from `--long-run DIR` for runs of your
own exported by `powerform.experiments.long_run_records`; the 80k ends always come from the
Table 1 records in `--results`.

    python -m powerform.experiments.training_curve [--results DIR] [--long-run DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from powerform.paths import OUTPUTS, RESULTS  # noqa: E402

OUT_DIR = OUTPUTS / "figures"
NAME = "figure2"
PDF_METADATA = {"CreationDate": None}

# arch, label, record stem, colour, x offset of the arm's hollow markers (steps)
ARMS = (
    ("physics", "PowerFORM", "case500_physics", "#2A6FBB", -2400),
    ("gridsfm", "GridSFM", "case500_gridsfm_decision", "#D9822B", 2400),
)
SEEDS = (0, 1, 2)
# Seeds side by side within an arm, so two seeds at nearly the same value (GridSFM's
# 0.201 and 0.202) still show as two markers. About half a marker width.
SEED_DX = 1200
SHORT, MID, LONG = 20000, 60000, 120000   # steps of the three budgets
BATCH = 4  # samples per step, checked against each record
MATCH_TOL = 1e-3
INK, MUTED = "#222222", "#6B7280"

# Applied on top of matplotlib's defaults around drawing and saving only, so neither this
# module nor another figure module imported first changes what the other draws.
RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
}


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"no record at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_end(path: Path, steps: int) -> float:
    """A run's cost at its end, after checking the record is a run of that budget."""
    run = load(path)
    if run["samples"] != BATCH * steps:
        raise SystemExit(f"{path.name}: {run['samples']} samples, expected {BATCH * steps}")
    return run["cost_mape"]


def curves(results: Path, long_run: Path | None = None) -> dict:
    """What the figure draws, per arm: run ends, the long run's milestones, its schedule."""
    long_run = Path(long_run) if long_run is not None else results / "long_run"
    out = {}
    for arch, _, stem, _, _ in ARMS:
        miles = load(long_run / f"milestones_{arch}_steps{LONG}_seed0.json")["rows"]
        line = {r["step"]: r["cost_mape"] for r in miles}
        end_long = run_end(long_run / f"{stem}_steps{LONG}_seed0.json", LONG)
        if abs(line[LONG] - end_long) > MATCH_TOL:
            raise SystemExit(f"{arch}: milestone {LONG} scores {line[LONG]}, the run recorded {end_long}")
        sched = load(long_run / f"curve_{arch}_steps{LONG}_seed0.json")["rows"]
        out[arch] = {
            "ends_short": [run_end(results / "table1" / f"{stem}_seed{s}.json", SHORT) for s in SEEDS],
            "end_mid": run_end(long_run / f"{stem}_steps{MID}_seed0.json", MID),
            "end_long": end_long,
            "line": dict(sorted(line.items())),
            "lr": {r["step"]: r["lr"] for r in sched},
        }
    if out["physics"]["lr"] != out["gridsfm"]["lr"]:
        raise SystemExit("the two long runs logged different schedules")
    return out


def build(results: Path = RESULTS, long_run: Path | None = None):
    """Draw the figure from the records; returns the open figure and what it drew. Save it
    with `save`, which applies the same style."""
    with plt.style.context(["default", RC]):
        return _draw(curves(Path(results), long_run))


def save(fig, out_dir: Path) -> None:
    with plt.style.context(["default", RC]):
        fig.savefig(out_dir / f"{NAME}.pdf", metadata=dict(PDF_METADATA))
        fig.savefig(out_dir / f"{NAME}.png", dpi=400)


def _draw(data: dict):
    fig, (ax, low) = plt.subplots(2, 1, figsize=(3.5, 1.8), sharex=True, height_ratios=[4, 1],
                                  gridspec_kw={"left": 0.125, "right": 0.76, "top": 0.98,
                                               "bottom": 0.205, "hspace": 0.12})
    finals = []
    for arch, label, _, colour, dx in ARMS:
        d = data[arch]
        xs = [SHORT + dx + SEED_DX * (s - 1) for s in SEEDS] + [MID + dx]
        ax.scatter(xs, d["ends_short"] + [d["end_mid"]], s=11, marker="o",
                   facecolors="white", edgecolors=colour, linewidths=0.7, zorder=4)
        ax.plot(list(d["line"]), list(d["line"].values()), color=colour, lw=1.0, marker="o", ms=2.6,
                mec="white", mew=0.3, label=f"{label}, {BATCH * LONG // 1000}k run", zorder=3,
                clip_on=False)
        finals.append((label, d["end_long"]))
    ax.scatter([], [], s=11, marker="o", facecolors="white", edgecolors=MUTED, linewidths=0.7,
               label=f"ends of {BATCH * SHORT // 1000}k, {BATCH * MID // 1000}k runs")

    ax.set_yscale("log")
    ax.set_ylim(0.05, 2.5)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2], labels=["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylabel("Cost MAPE (%)")
    ax.grid(axis="y", color="#E5E7EB", lw=0.4, zorder=0)
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.1, labelspacing=0.15)
    finals.sort(key=lambda f: f[1])
    for i, (label, y) in enumerate(finals):
        dy = 0 if len(finals) == 1 else (-3.5 if i == 0 else 3.5)
        ax.annotate(f"{label} {y:.3f}", (LONG, y), xytext=(4, dy), textcoords="offset points",
                    va="center", fontsize=7, color=INK, annotation_clip=False)

    lr = data["physics"]["lr"]
    low.plot(list(lr), list(lr.values()), color=MUTED, lw=1.0)
    low.set_ylim(0, 3.3e-4)
    low.set_yticks([0, 3e-4], labels=["0", r"3$\times$10$^{-4}$"])
    low.text(0.99, 0.9, "learning rate", transform=low.transAxes, ha="right", va="top",
             fontsize=7, color=MUTED)
    low.set_xlim(0, LONG)
    low.set_xticks(range(0, LONG + 1, 20000), labels=[str(BATCH * s // 1000) for s in range(0, LONG + 1, 20000)])
    low.set_xlabel(r"Training samples ($\times 10^3$)")
    return fig, data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results tree holding table1/ and long_run/")
    ap.add_argument("--long-run", type=Path, default=None,
                    help="long-run records to draw instead of <results>/long_run")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig, data = build(a.results, a.long_run)
    save(fig, a.out_dir)
    plt.close(fig)
    for arch, label, *_ in ARMS:
        d = data[arch]
        print(f"{label}: ends {BATCH * SHORT // 1000}k {' '.join(f'{e:.3f}' for e in d['ends_short'])}, "
              f"{BATCH * MID // 1000}k {d['end_mid']:.3f}, {BATCH * LONG // 1000}k {d['end_long']:.3f}")
    print(f"-> {a.out_dir / NAME}.pdf, .png")


if __name__ == "__main__":
    main()
