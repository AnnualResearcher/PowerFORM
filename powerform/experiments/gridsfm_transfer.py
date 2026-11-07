"""App. D, reconfiguration with the GridSFM backbone: the settled-step table and Figure 3.

The table gives, for each backbone, the step after which a seed's infeasible fraction
never again exceeds 0.5, as the median over seeds, for the pretrained and the scratch
arm. Figure 3 draws the GridSFM records in the format of Figure 1: min-max bands over
the three seeds, seed-mean line, nothing smoothed, the same gap clip.

    python -m powerform.experiments.gridsfm_transfer [--results DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from powerform.experiments import figure  # noqa: E402
from powerform.paths import OUTPUTS, RESULTS  # noqa: E402

FIG_DIR = OUTPUTS / "figures"
TABLE_DIR = OUTPUTS / "reconfiguration"
NAME = "figure3"
TABLE = "settled_steps.json"
THRESHOLD = 0.5
FAMILIES = (("physics", "PowerFORM"), ("generic", "PowerFORM, generic attn."),
            ("gridsfm", "GridSFM + attached gate"))


def settled_step(rows: list[dict], threshold: float = THRESHOLD) -> int | None:
    """The step from which a run's infeasible fraction never again exceeds `threshold`;
    None if its last evaluation is still above it."""
    curve = {r["step"]: r["value"] for r in rows if r["metric"] == "infeasible_fraction"}
    above = [t for t, v in curve.items() if v > threshold]
    after = [t for t in curve if not above or t > max(above)]
    return min(after) if after else None


def table(results: Path = RESULTS) -> dict:
    """The settled-step table from the reconfiguration records in `results`."""
    out = {"threshold": THRESHOLD, "rows": []}
    for maps, label in FAMILIES:
        seeds = figure.complete_seeds(results, maps)
        row = {"maps": maps, "backbone": label, "seeds": [int(s) for s in seeds]}
        for arm in figure.ARMS:
            runs = [json.loads((results / "reconfiguration" / f"{maps}_{arm}_seed{s}.json")
                               .read_text(encoding="utf-8")) for s in seeds]
            steps = [settled_step(r["rows"]) for r in runs]
            if None in steps:
                raise SystemExit(f"{maps} {arm}: a seed never settles below {THRESHOLD}")
            row[arm] = {"per_seed": steps, "median": int(statistics.median(steps))}
            if arm == "pretrained":
                row["starting_models"] = len({r["init"] for r in runs})
        out["rows"].append(row)
    return out


def build(results: Path = RESULTS):
    """Draw Figure 3 from the GridSFM records in `results`; the figure is left open."""
    results = Path(results)
    fig = plt.figure(figsize=(3.5, 1.8))
    gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.13, right=0.97, top=figure.TOP, bottom=0.36)
    ax = [fig.add_subplot(gs[0, j]) for j in range(2)]
    seeds = figure.complete_seeds(results, "gridsfm")
    for arm, st in figure.ARMS.items():
        gap, inf = figure.load(results, "gridsfm", arm, seeds)
        figure.draw(ax[0], gap, st, st["label"], clip=(figure.A_YMIN, figure.A_YMAX))
        figure.draw(ax[1], inf, st, None, clip=None)
    # each panel keeps its own y labels: the pair does not share an axis
    figure.style_axes(ax[0], 0.0, (figure.A_YMIN, figure.A_YMAX), first_of_pair=True)
    figure.style_axes(ax[1], figure.random_infeasible_rate(results), (-0.03, 1.03), first_of_pair=True)
    for a, title in zip(ax, ("(a) eval gap", "(b) infeasible fraction")):
        a.set_title(title, loc="left", fontsize=8, pad=2)
    ax[0].set_ylabel("eval gap (%)")
    ax[1].set_ylabel("infeasible fraction")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.55, 0.0), handlelength=1.6, columnspacing=1.4)
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="results tree holding reconfiguration/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default outputs/figures for the figure, outputs/reconfiguration for the table")
    a = ap.parse_args(argv)
    fig_dir = a.out_dir or FIG_DIR
    table_dir = a.out_dir or TABLE_DIR
    for d in (fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    t = table(a.results)
    (table_dir / TABLE).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(f"settled step (infeasible fraction stays <= {THRESHOLD}), median over seeds:")
    for row in t["rows"]:
        print(f"  {row['backbone']:<26} seeds {len(row['seeds'])}  pretrained "
              f"{row['pretrained']['median']:>6,}  scratch {row['scratch']['median']:>6,}  "
              f"(starting models {row['starting_models']})")
    fig = build(a.results)
    fig.savefig(fig_dir / f"{NAME}.pdf", metadata=dict(figure.PDF_METADATA))
    fig.savefig(fig_dir / f"{NAME}.png", dpi=400)
    plt.close(fig)
    print(f"-> {table_dir / TABLE}, {fig_dir / NAME}.pdf, .png")


if __name__ == "__main__":
    main()
