"""The reconfiguration study: Figure 1, its reductions and every number in its text.

From the shipped records alone: the plotted arrays against a recomputation, the
reductions against their records, the paper's numbers, the task settings against the
trainer's defaults. With data and checkpoints present: the outages, the first rows of
the reference and the infeasible split are regenerated.

    python powerform/tests/test_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from powerform.data.opfdata import fetch_command, shard_path  # noqa: E402
from powerform.eval.checkpoint import load_model  # noqa: E402
from powerform.eval.infeasible_split import split_of  # noqa: E402
from powerform.experiments import figure, seed_table, windows  # noqa: E402
from powerform.experiments.reference import N_SCENARIOS, reference_table, task_pools  # noqa: E402
from powerform.model.line_track import LineStatusTrack  # noqa: E402
from powerform.paths import CHECKPOINTS, RESULTS  # noqa: E402
from powerform.training import reconfiguration as R  # noqa: E402

REC = RESULTS / "reconfiguration"
SEEDS = {"physics": list(range(6)), "generic": list(range(3))}
FETCH_CKPT = "python scripts/download_checkpoints.py --group reconfig"
REL_TOL, ABS_FLOOR = 1e-10, 1e-12


class Skip(Exception):
    pass


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def perturbed_copy(dst: Path) -> Path:
    """The records with one infeasible fraction moved by one scored scenario."""
    shutil.copytree(REC, dst / "reconfiguration")
    f = dst / "reconfiguration" / "physics_pretrained_seed3.json"
    d = load(f)
    r = next(x for x in d["rows"] if x["metric"] == "infeasible_fraction" and x["step"] == 400)
    r["value"] = r["value"] + 1 / 85 if r["value"] + 1 / 85 <= 1 else r["value"] - 1 / 85
    f.write_text(json.dumps(d))
    return dst


def plotted_vs_records(fig, results: Path) -> list[str]:
    """The figure's mean lines and min-max bands against the run records."""
    bad = []
    probe = plt.figure().add_subplot()
    for fi, (maps, _) in enumerate(figure.FAMILIES):
        for arm, style in figure.ARMS.items():
            curves = {"gap_pct_median": defaultdict(list), "infeasible_fraction": defaultdict(list)}
            for s in SEEDS[maps]:
                for r in load(results / "reconfiguration" / f"{maps}_{arm}_seed{s}.json")["rows"]:
                    if r["metric"] in curves:
                        curves[r["metric"]][r["step"]].append(r["value"])
            for metric, j, clip in (("gap_pct_median", fi, (figure.A_YMIN, figure.A_YMAX)),
                                    ("infeasible_fraction", 2 + fi, None)):
                per = curves[metric]
                steps = np.array(sorted(per))
                mean = np.array([np.mean(per[t]) for t in steps])
                lo = np.array([np.min(per[t]) for t in steps])
                hi = np.array([np.max(per[t]) for t in steps])
                if clip is not None:
                    lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
                    mean = np.where(mean > clip[1], np.nan, mean)
                ax = fig.axes[j]
                line = [ln for ln in ax.get_lines()
                        if ln.get_color() == style["color"] and ln.get_marker() == "None"]
                if len(line) != 1 or not np.array_equal(line[0].get_ydata(), mean, equal_nan=True):
                    bad.append(f"{maps} {arm} {metric}: line")
                want = probe.fill_between(steps, lo, hi).get_paths()[0].vertices
                bands = [c for c in ax.collections
                         if np.allclose(c.get_facecolor()[0][:3], matplotlib.colors.to_rgb(style["color"]))]
                if len(bands) != 1 or not np.array_equal(bands[0].get_paths()[0].vertices, want):
                    bad.append(f"{maps} {arm} {metric}: band")
    plt.close(probe.figure)
    return bad


def check_figure() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fig = figure.build(RESULTS)
        assert not plotted_vs_records(fig, RESULTS)
        plt.close(fig)
        figure.main(["--results", str(RESULTS), "--out-dir", str(tmp / "a")])
        figure.main(["--results", str(RESULTS), "--out-dir", str(tmp / "b")])
        for ext in ("png", "pdf"):
            assert (tmp / "a" / f"figure1.{ext}").read_bytes() == (tmp / "b" / f"figure1.{ext}").read_bytes()
            same = (tmp / "a" / f"figure1.{ext}").read_bytes() == (RESULTS / "figures" / f"figure1.{ext}").read_bytes()
            print(f"  figure1.{ext} {'equals' if same else 'differs from'} the shipped file "
                  "(fonts and library versions decide this)")
        bad = perturbed_copy(tmp / "bad")
        fig = figure.build(bad)
        d = plotted_vs_records(fig, RESULTS)
        plt.close(fig)
        assert d, "the comparison missed a moved value"

        ref = tmp / "ref"
        shutil.copytree(REC, ref / "reconfiguration")
        p = ref / "reconfiguration" / "reference_val.json"
        doc = load(p)
        row = next(r for r in doc["rows"] if r["ref_pu"] is not None and r["n_random_converged"])
        row["n_random_converged"] = 0
        p.write_text(json.dumps(doc))
        fig = figure.build(ref)
        ys = [ln.get_ydata()[0] for ln in fig.axes[2].get_lines() if ln.get_color() == "0.45"]
        plt.close(fig)
        assert ys == [figure.random_infeasible_rate(ref)] != [figure.random_infeasible_rate(RESULTS)]
    print("lines and bands equal the records, two runs write the same bytes, a moved "
          "record and a moved reference both show")


def check_reductions() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        windows.main(["--results", str(RESULTS), "--out-dir", str(tmp / "new")])
        seed_table.main(["--results", str(RESULTS), "--out-dir", str(tmp / "new")])
        names = [f"{kind}_{maps}.json" for kind in ("windows", "crossings", "seed_table")
                 for maps in ("physics", "generic")]
        for name in names:
            assert load(tmp / "new" / name) == load(REC / name), name
        bad = perturbed_copy(tmp / "bad")
        windows.main(["--results", str(bad), "--out-dir", str(tmp / "out")])
        assert load(tmp / "out" / "windows_physics.json") != load(REC / "windows_physics.json")
        seed_table.main(["--results", str(bad), "--out-dir", str(tmp / "out")])
        assert load(tmp / "out" / "seed_table_physics.json") != load(REC / "seed_table_physics.json")
    print(f"{len(names)} reductions regenerate exactly; a moved record changes them")


def check_paper_numbers() -> None:
    cross = load(REC / "crossings_physics.json")["first_crossing"]
    assert cross["infeasible_fraction"]["pretrained"] == 500
    assert cross["infeasible_fraction"]["scratch"] == 2250
    assert cross["gap_pct_median"]["pretrained"] == 700
    assert cross["gap_pct_median"]["scratch"] == 2500
    assert load(REC / "crossings_generic.json")["first_crossing"]["infeasible_fraction"]["scratch"] is None

    split = {arm: [load(REC / f"infeasible_split_physics_{arm}_seed{s}.json") for s in range(6)]
             for arm in ("pretrained", "scratch")}
    share = {arm: sum(r["diverged"] for r in rows) / sum(r["n_rejected"] for r in rows)
             for arm, rows in split.items()}
    assert f"{share['pretrained'] * 100:.1f}" == "1.2" and f"{share['scratch'] * 100:.1f}" == "2.8", share
    run = load(REC / "physics_pretrained_seed0.json")
    assert f"{run['controls']['all_closed_gap_pct']:+.2f}" == "+0.15"
    base = load(REC / "base_feasibility.json")["summary"]
    assert base["base_converges"] == 157 and base["base_converges_but_out_of_box"] / 157 > 0.685
    assert f"{base['base_converges_but_out_of_box'] / 157 * 100:.0f}" == "69"
    rows = load(REC / "reference_val.json")["rows"]
    scored = [r for r in rows if r["ref_pu"] is not None]
    assert (len(rows), len(scored), len(rows) - len(scored)) == (157, 85, 72)
    assert f"{sum(not r['n_random_converged'] for r in scored) / 85 * 100:.0f}" == "79"
    assert f"{load(REC / 'islanding_prior.json')['disconnect_fraction'] * 100:.1f}" == "15.4"
    steps = sorted({r["step"] for r in run["rows"] if r["metric"] == "gap_pct"})
    assert len(steps) == 51 and steps == sorted(R.EVAL_STEPS)
    track = LineStatusTrack(216, 8, 32)
    assert sum(p.numel() for p in track.parameters()) == 332321
    print("crossings, rejection shares, baselines, usable scenarios, eval steps and "
          "the track's size equal the text")


def check_settings_are_the_trainers_defaults() -> None:
    argv = sys.argv
    sys.argv = ["x", "--maps", "physics", "--arm", "scratch", "--seed", "0"]
    try:
        args = R.parse_args()
    finally:
        sys.argv = argv
    for maps, seeds in SEEDS.items():
        for arm in ("pretrained", "scratch"):
            for s in seeds:
                d = load(REC / f"{maps}_{arm}_seed{s}.json")
                for field in ("steps", "n_train", "batch", "T", "gumbel_tau", "margin", "lr",
                              "trunk_lr_mult", "k_open", "z_track"):
                    assert d[field] == getattr(args, field), (maps, arm, s, field)
                want = None if arm == "scratch" else R.default_init(maps, s).name
                assert d["init"] == want, (maps, arm, s, d["init"])
    assert R.default_init("physics", 3).name == "opf_generalist_physics_ed_seed0.pt"
    assert R.default_init("physics", 2).name == "opf_generalist_physics_ed_seed2.pt"
    print("every run record carries the trainer's defaults and its default init")


def check_label_rules() -> None:
    logits = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    assert float(R.ranking_hinge(logits, (0, 1), 0.5)) == 0.0
    assert abs(float(R.ranking_hinge(logits, (0, 2), 0.5)) - 1.5) < 1e-12
    rng = np.random.default_rng(0)
    cands = R.candidates(logits, 2, 4, 2.0, rng, (2, 3))
    assert (0, 1) in cands and (2, 3) in cands and len(cands) == len(set(cands))
    print("the hinge is zero past the margin; candidates hold the greedy and the incumbent")


def need_task(*ckpt: str) -> None:
    if not shard_path("case57", "fulltop", "val").exists():
        raise Skip(f"case57 absent; fetch with `{fetch_command('case57', 'fulltop')}`")
    for name in ckpt:
        if not (CHECKPOINTS / name).exists():
            raise Skip(f"{name} absent; fetch with `{FETCH_CKPT}`")


def check_reference_regenerates() -> None:
    need_task()
    _, pool_cpu, outages = task_pools("val", N_SCENARIOS["val"], 12, torch.device("cpu"))
    doc = load(REC / "reference_val.json")
    assert outages == doc["outage_lines"], "the outage draw moved"
    pool_cpu.insts = pool_cpu.insts[:4]
    rows = reference_table(pool_cpu, "val", doc["k"], outages[:4])["rows"]
    for got, want in zip(rows, doc["rows"][:4]):
        assert got["ref_action"] == want["ref_action"] and got["n_random_converged"] == want["n_random_converged"]
        for k in ("base_pu", "ref_pu", "random_mean_pu"):
            a, b = got[k], want[k]
            assert (a is None and b is None) or abs(a - b) <= max(ABS_FLOOR, REL_TOL * abs(b)), (k, a, b)
    g = pool_cpu.insts[0].tgrid
    g.p_load = g.p_load.clone()
    g.p_load[int(torch.argmax(g.p_load.abs()))] += 1e-7
    moved = reference_table(pool_cpu, "val", doc["k"], outages[:4])["rows"][0]
    assert moved["base_pu"] != rows[0]["base_pu"], "a moved load went unnoticed"
    print("157 outages and the first 4 reference rows regenerate; a moved load shows")


def check_split_regenerates() -> None:
    name = "reconfig_physics_pretrained_seed0.pt"
    need_task(name)
    net = load_model(CHECKPOINTS / name, "cpu")
    pool_dev, pool_cpu, _ = task_pools("val", N_SCENARIOS["val"], net.trunk.n_cost_feats,
                                       torch.device("cpu"))
    got = split_of(net, pool_dev, pool_cpu, load(REC / "reference_val.json"))
    want = load(REC / "infeasible_split_physics_pretrained_seed0.json")
    assert all(got[k] == want[k] for k in got), (got, want)
    print(f"pretrained seed 0: ok {got['ok']}, diverged {got['diverged']}, out of box "
          f"{got['out_of_box']}, record agrees")


CHECKS = (check_figure, check_reductions, check_paper_numbers,
          check_settings_are_the_trainers_defaults, check_label_rules,
          check_reference_regenerates, check_split_regenerates)


def main() -> None:
    ran, failed, skipped = 0, 0, []
    for check in CHECKS:
        name = check.__name__[len("check_"):]
        try:
            check()
            ran += 1
            print(f"[PASS] {name}", flush=True)
        except Skip as e:
            skipped.append(str(e))
            print(f"[SKIP] {name}: {e}", flush=True)
        except (Exception, SystemExit):  # noqa: BLE001 -- reported, and the exit code says so
            ran += 1
            failed += 1
            traceback.print_exc()
            print(f"[FAIL] {name}", flush=True)
    why = f" ({'; '.join(dict.fromkeys(skipped))})" if skipped else ""
    print(f"{ran} checks ran, {failed} failed, {len(skipped)} skipped{why}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
