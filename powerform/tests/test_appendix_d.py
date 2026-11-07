"""App. D: longer training (Figure 2) and reconfiguration with the GridSFM backbone
(Figure 3 and its table).

From the shipped records alone: the records are the runs the text names, both figures
and the table regenerate from them, a moved value shows, and the text's numbers hold.
Without data: the GridSFM attachment's size and initialisation. With case57 present:
the attachment leaves GridSFM unchanged at start and its gate acts. With checkpoints and
data present: spot re-scores of the long runs and of a GridSFM reconfiguration model.

    python powerform/tests/test_appendix_d.py
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

from powerform.data.corpus.datakit_export import CORPUS_DIR, DATAKIT_HOME  # noqa: E402
from powerform.data.opfdata import fetch_command, shard_path  # noqa: E402
from powerform.experiments import figure, gridsfm_transfer, training_curve  # noqa: E402
from powerform.paths import CHECKPOINTS, RESULTS  # noqa: E402

torch.set_num_threads(1)
LONG = RESULTS / "long_run"
REC = RESULTS / "reconfiguration"
BATCH = 4
STEPS = (60000, 120000)
STEMS = {"physics": "case500_physics", "gridsfm": "case500_gridsfm_decision"}
HOLD_OUT, N_TRAIN = 500, 29487
GENERALIST = "opf_generalist_gridsfm_seed0.pt"
GRIDSFM_RECONFIG_PARAMS = 15168543
FETCH_LONG = "python scripts/download_checkpoints.py --group long_run"
FETCH_RECONFIG = "python scripts/download_checkpoints.py --group gridsfm_reconfig"
TOL_F64 = 1e-9   # relative, float64 PowerFORM re-scores
TOL_PP = 1e-3    # percentage points of cost MAPE, float32 GridSFM re-scores of GPU-recorded runs


class Skip(Exception):
    pass


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_bytes_twice(main, name: str, tmp: Path) -> None:
    """Two runs on the same records write the same bytes."""
    for d in ("a", "b"):
        main(["--results", str(RESULTS), "--out-dir", str(tmp / d)])
    for ext in ("png", "pdf"):
        a, b = (tmp / d / f"{name}.{ext}" for d in ("a", "b"))
        assert a.read_bytes() == b.read_bytes(), f"{name}.{ext} differs between two runs"
        same = a.read_bytes() == (RESULTS / "figures" / f"{name}.{ext}").read_bytes()
        print(f"  {name}.{ext} {'equals' if same else 'differs from'} the shipped file "
              "(fonts and library versions decide this)")


def check_long_run_records() -> None:
    ends = {}
    for arch, stem in STEMS.items():
        for steps in STEPS:
            r = load(LONG / f"{stem}_steps{steps}_seed0.json")
            assert (r["steps"], r["samples"], r["seed"]) == (steps, BATCH * steps, 0), (arch, steps)
            assert r["n_train"] == N_TRAIN and r["n_nonconverged"] == 0, (arch, steps)
            assert len(r["per_scenario_cost_mape"]) == HOLD_OUT, (arch, steps)
            assert abs(np.mean(r["per_scenario_cost_mape"]) - r["cost_mape"]) < 1e-12, (arch, steps)
            ends[arch, steps] = r["cost_mape"]
        miles = load(LONG / f"milestones_{arch}_steps120000_seed0.json")["rows"]
        assert [m["step"] for m in miles] == list(range(5000, 120001, 5000)), arch
        assert all(m["n_nonconverged"] == 0 for m in miles), arch
        assert abs(miles[-1]["cost_mape"] - ends[arch, 120000]) <= 1e-3, arch
        curve = load(LONG / f"curve_{arch}_steps120000_seed0.json")["rows"]
        assert [c["step"] for c in curve] == list(range(200, 120001, 200)), arch
        assert abs(curve[0]["lr"] - 3e-4) < 1e-6 and abs(curve[-1]["lr"] - 1e-5) < 1e-12, arch
    # The text: values, and the saturation reading (10% lower or not).
    printed = {k: f"{v:.3f}" for k, v in ends.items()}
    assert printed == {("physics", 60000): "0.084", ("physics", 120000): "0.070",
                       ("gridsfm", 60000): "0.098", ("gridsfm", 120000): "0.106"}, printed
    assert ends["gridsfm", 120000] >= ends["gridsfm", 60000]
    assert ends["physics", 120000] < 0.9 * ends["physics", 60000]
    short = [load(RESULTS / "table1" / f"case500_physics_seed{s}.json")["cost_mape"] for s in range(3)]
    assert (f"{min(short):.3f}", f"{max(short):.3f}") == ("0.095", "0.198")
    print("four run ends, 2 x 24 milestones and 2 x 600 log rows are the runs the text names; "
          "0.084, 0.070, 0.098, 0.106 and the 0.095-0.198 spread hold")


def fake_runs(root: Path) -> None:
    """Run folders as `long_run` and `milestones` leave them, rebuilt from the shipped records:
    the run record, milestone scores as JSON lines split over two scoring processes, the log."""
    for arch, stem in STEMS.items():
        for steps in STEPS:
            work = root / f"steps{steps}"
            (work / "out").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LONG / f"{stem}_steps{steps}_seed0.json", work / "out" / f"{stem}_seed0.json")
        work = root / "steps120000"
        rows = load(LONG / f"milestones_{arch}_steps120000_seed0.json")["rows"]
        lines = [json.dumps({"name": f"step{r['step']:06d}", **{k: v for k, v in r.items() if k != "step"}})
                 for r in reversed(rows)]
        for k in range(2):
            (work / "out" / f"milestones_{stem}_seed0.part{k}.jsonl").write_text(
                "".join(f"{line}\n" for line in lines[k::2]))
        (work / "curves").mkdir(exist_ok=True)
        curve = load(LONG / f"curve_{arch}_steps120000_seed0.json")["rows"]
        (work / "curves" / f"{stem}_seed0.jsonl").write_text("".join(f"{json.dumps(r)}\n" for r in curve))


def check_long_run_records_export() -> None:
    from powerform.experiments import long_run_records
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake_runs(tmp / "runs")
        for arch in STEMS:
            long_run_records.main(["--arch", arch, "--steps", *map(str, STEPS),
                                   "--work-root", str(tmp / "runs"), "--out-dir", str(tmp / "out")])
        made = sorted(p.name for p in (tmp / "out").iterdir())
        assert made == sorted(p.name for p in LONG.iterdir()), made
        for name in made:
            assert (tmp / "out" / name).read_bytes() == (LONG / name).read_bytes(), name
        part = tmp / "runs" / "steps120000" / "out" / "milestones_case500_physics_seed0.part0.jsonl"
        part.write_text(part.read_text() + part.read_text().splitlines()[0] + "\n")
        try:
            long_run_records.main(["--arch", "physics", "--steps", "120000",
                                   "--work-root", str(tmp / "runs"), "--out-dir", str(tmp / "dup")])
        except SystemExit:
            pass
        else:
            raise AssertionError("a milestone scored twice went unnoticed")
    print(f"run folders export to the {len(made)} shipped long_run files byte for byte; "
          "a milestone scored twice is refused")


def training_curve_vs_records(fig, data: dict) -> list[str]:
    """The drawn lines and hollow markers against what `curves` read."""
    ax = fig.axes[0]
    bad = []
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]
    offsets = [c.get_offsets() for c in ax.collections if len(c.get_offsets())]
    if len(lines) != len(training_curve.ARMS) or len(offsets) != len(training_curve.ARMS):
        return [f"{len(lines)} lines and {len(offsets)} marker sets drawn"]
    for (arch, *_), ln in zip(training_curve.ARMS, lines):
        if list(ln.get_ydata()) != list(data[arch]["line"].values()):
            bad.append(f"{arch} line")
    for (arch, *_), off in zip(training_curve.ARMS, offsets):
        want = data[arch]["ends_short"] + [data[arch]["end_mid"]]
        if list(np.asarray(off)[:, 1]) != want:
            bad.append(f"{arch} markers")
    return bad


def check_training_curve() -> None:
    fig, data = training_curve.build(RESULTS)
    assert not training_curve_vs_records(fig, data)
    for arch in ("physics", "gridsfm"):
        miles = load(LONG / f"milestones_{arch}_steps120000_seed0.json")["rows"]
        assert data[arch]["line"] == {m["step"]: m["cost_mape"] for m in miles}, arch
    plt.close(fig)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        same_bytes_twice(training_curve.main, training_curve.NAME, tmp)
        bad = tmp / "bad"
        shutil.copytree(RESULTS / "table1", bad / "table1")
        shutil.copytree(LONG, bad / "long_run")
        p = bad / "long_run" / "milestones_gridsfm_steps120000_seed0.json"
        doc = load(p)
        doc["rows"][10]["cost_mape"] *= 1.5
        p.write_text(json.dumps(doc))
        moved, _ = training_curve.build(bad)
        assert training_curve_vs_records(moved, data), "a moved milestone went unnoticed"
        plt.close(moved)
        p = bad / "table1" / "case500_gridsfm_decision_seed1.json"
        doc = load(p)
        doc["cost_mape"] *= 1.5
        p.write_text(json.dumps(doc))
        moved, _ = training_curve.build(bad, LONG)
        assert training_curve_vs_records(moved, data), "a moved 80k end went unnoticed"
        plt.close(moved)
        p = bad / "long_run" / "case500_physics_steps120000_seed0.json"
        doc = load(p)
        doc["cost_mape"] += 0.01
        p.write_text(json.dumps(doc))
        try:
            training_curve.curves(RESULTS, bad / "long_run")
        except SystemExit:
            pass
        else:
            raise AssertionError("a run end that its last milestone does not reproduce went unnoticed")
        same, _ = training_curve.build(RESULTS, LONG)
        assert not training_curve_vs_records(same, data), "--long-run results/long_run differs"
        plt.close(same)
    print("lines and markers equal the records, two runs write the same bytes; a moved "
          "milestone, a moved 80k end and a run end its last milestone misses all show")


def check_settled_table() -> None:
    t = gridsfm_transfer.table(RESULTS)
    assert t == load(REC / gridsfm_transfer.TABLE), "the table does not regenerate"
    got = {r["maps"]: (len(r["seeds"]), r["pretrained"]["median"], r["scratch"]["median"],
                       r["starting_models"]) for r in t["rows"]}
    assert got == {"physics": (6, 400, 1875, 3), "generic": (3, 950, 19000, 3),
                   "gridsfm": (3, 19000, 2250, 1)}, got
    g = next(r for r in t["rows"] if r["maps"] == "gridsfm")
    assert g["pretrained"]["per_seed"] == [16000, 20000, 19000]
    assert g["scratch"]["per_seed"] == [8000, 2250, 1000]
    assert all(p > s for p, s in zip(g["pretrained"]["per_seed"], g["scratch"]["per_seed"]))
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td)
        shutil.copytree(REC, bad / "reconfiguration")
        p = bad / "reconfiguration" / "gridsfm_scratch_seed1.json"
        doc = load(p)
        r = next(x for x in doc["rows"] if x["metric"] == "infeasible_fraction" and x["step"] == 5000)
        r["value"] = 0.9
        p.write_text(json.dumps(doc))
        assert gridsfm_transfer.table(bad) != t, "a moved infeasible fraction went unnoticed"
    print("the table regenerates; 400/1,875, 950/19,000, 19,000/2,250, the per-seed steps "
          "and the starting-model counts equal the text; a moved value shows")


def figure3_vs_records(fig, results: Path) -> list[str]:
    bad = []
    for j, metric in ((0, "gap_pct_median"), (1, "infeasible_fraction")):
        # mean lines are drawn with a line style; the off-scale edge triangles without one
        lines = [ln for ln in fig.axes[j].get_lines()
                 if ln.get_linestyle() not in ("None", " ", "") and len(ln.get_xdata()) > 2]
        if len(lines) != len(figure.ARMS):
            bad.append(f"{metric}: {len(lines)} mean lines drawn")
            continue
        for (arm, _), ln in zip(figure.ARMS.items(), lines):
            per = defaultdict(list)
            for s in range(3):
                for r in load(results / "reconfiguration" / f"gridsfm_{arm}_seed{s}.json")["rows"]:
                    if r["metric"] == metric:
                        per[r["step"]].append(r["value"])
            steps = sorted(per)
            mean = np.array([np.mean(per[t]) for t in steps])
            if metric == "gap_pct_median":
                mean = np.where(mean > figure.A_YMAX, np.nan, mean)
            got = np.asarray(ln.get_ydata(), dtype=float)
            if list(ln.get_xdata()) != steps or not np.array_equal(np.isnan(got), np.isnan(mean)) \
                    or not np.allclose(got[~np.isnan(got)], mean[~np.isnan(mean)], rtol=0, atol=1e-12):
                bad.append(f"{metric} {arm}")
    return bad


def check_figure3() -> None:
    fig = gridsfm_transfer.build(RESULTS)
    assert not figure3_vs_records(fig, RESULTS)
    plt.close(fig)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for d in ("a", "b"):
            gridsfm_transfer.main(["--results", str(RESULTS), "--out-dir", str(tmp / d)])
        for ext in ("png", "pdf"):
            a, b = (tmp / d / f"{gridsfm_transfer.NAME}.{ext}" for d in ("a", "b"))
            assert a.read_bytes() == b.read_bytes()
            same = a.read_bytes() == (RESULTS / "figures" / f"{gridsfm_transfer.NAME}.{ext}").read_bytes()
            print(f"  {gridsfm_transfer.NAME}.{ext} {'equals' if same else 'differs from'} the "
                  "shipped file (fonts and library versions decide this)")
        bad = tmp / "bad"
        shutil.copytree(REC, bad / "reconfiguration")
        p = bad / "reconfiguration" / "gridsfm_pretrained_seed2.json"
        doc = load(p)
        r = next(x for x in doc["rows"] if x["metric"] == "infeasible_fraction" and x["step"] == 400)
        r["value"] = r["value"] - 1 / 85 if r["value"] >= 1 / 85 else r["value"] + 1 / 85
        p.write_text(json.dumps(doc))
        moved = gridsfm_transfer.build(bad)
        assert figure3_vs_records(moved, RESULTS), "a moved value went unnoticed"
        plt.close(moved)
    print("lines equal the records, two runs write the same bytes, a moved value shows")


def check_gridsfm_settings() -> None:
    from powerform.training import reconfiguration as R
    from powerform.training import reconfiguration_gridsfm as G
    args = G.parse_args(["--arm", "scratch", "--seed", "0"])
    argv, sys.argv = sys.argv, ["x", "--maps", "physics", "--arm", "scratch", "--seed", "0"]
    try:
        release = R.parse_args()
    finally:
        sys.argv = argv
    for field in R.RESUME_MUST_MATCH:
        if field != "maps":
            assert getattr(args, field) == getattr(release, field), f"default {field} drifted"
    physics = load(REC / "physics_scratch_seed0.json")
    for arm in ("pretrained", "scratch"):
        for s in range(3):
            d = load(REC / f"gridsfm_{arm}_seed{s}.json")
            assert (d["maps"], d["arm"], d["seed"], d["params"]) == ("gridsfm", arm, s, GRIDSFM_RECONFIG_PARAMS)
            for field in ("steps", "n_train", "batch", "T", "gumbel_tau", "margin", "lr",
                          "trunk_lr_mult", "k_open", "z_track"):
                assert d[field] == getattr(args, field) == physics[field], (arm, s, field)
            for field in ("grid", "n_val", "n_scored", "controls"):
                assert d[field] == physics[field], (arm, s, field)
            assert d["init"] == (None if arm == "scratch" else GENERALIST), (arm, s, d["init"])
    print("every GridSFM record carries the trainer's defaults, the physics study's settings "
          "and controls, and one shared generalist")


def check_gridsfm_attachment() -> None:
    from powerform.model.baselines.gridsfm_track import GatedConv, build
    torch.manual_seed(0)
    net = build("scratch", 32)
    assert sum(p.numel() for p in net.parameters()) == GRIDSFM_RECONFIG_PARAMS
    track = net.trunk.z_stream
    assert len(track.updates) == net.trunk.num_blocks == 8
    assert all(u.in_features == 3 * 128 + 1 == 385 and not u.weight.any() and not u.bias.any()
               for u in track.updates)
    assert all(isinstance(b.gnn, GatedConv) for b in net.trunk.blocks)
    print("15,168,543 parameters, a zero-initialised 385 -> 32 update in each of the 8 blocks, "
          "every block's message passing wrapped")


def need_case57() -> None:
    if not shard_path("case57", "fulltop", "val").exists():
        raise Skip(f"case57 absent; fetch with `{fetch_command('case57', 'fulltop')}`")


def gridsfm_pool(n: int):
    from powerform.data.binding import reconfiguration_binding
    from powerform.experiments.reference import GRID, N_COST_FEATS, task_pools
    from powerform.model.baselines.gridsfm_track import attach_graphs
    pool_dev, pool_cpu, outages = task_pools("val", n, N_COST_FEATS, torch.device("cpu"))
    attach_graphs(pool_dev, GRID, "val", outages)
    return pool_dev, pool_cpu, reconfiguration_binding


def check_gridsfm_identity_and_gate() -> None:
    need_case57()
    from powerform.model.baselines.gridsfm_track import build
    from powerform.third_party.gridsfm.model import GridTransformerBackbone

    class Stock(GridTransformerBackbone):
        def _predict_and_write(self, data, h_grid_bus, h_grid_gen, h_grid_global, n_graphs):
            self.h_bus = h_grid_bus

    pool_dev, _, bind = gridsfm_pool(2)
    torch.manual_seed(0)
    net = build("scratch", 32).eval()
    inst, spec = bind(pool_dev.insts[0], pool_dev.labels[0], pool_dev.specs[0])
    d = net.graph(inst, spec)
    with torch.no_grad():
        ell = net.trunk.line_logits(d.clone())
        h_gated = net.trunk.h_bus.clone()
        stock = Stock()
        state = {k.replace(".gnn.inner.", ".gnn."): v for k, v in net.trunk.state_dict().items()
                 if not k.startswith("z_stream")}
        missing, unexpected = stock.load_state_dict(state, strict=False)
        assert not unexpected and all(m.startswith("head_") for m in missing), (missing, unexpected)
        stock.float().eval()(d.clone())
        assert torch.all(ell == 0) and torch.equal(stock.h_bus, h_gated), "not GridSFM's own forward at start"

        ell0 = torch.zeros_like(ell)
        net.trunk.line_logits(d.clone(), ell0)
        assert torch.equal(net.trunk.h_bus, h_gated)
        ell0[5] = -30.0                     # gate exp(-30): line 5 closed
        net.trunk.line_logits(d.clone(), ell0)
        u, v = int(net.trunk.state.u[5]), int(net.trunk.state.v[5])
        moved = (net.trunk.h_bus - h_gated).abs().amax(dim=-1)
        assert moved[u] > 1e-3 and moved[v] > 1e-3, "closing a gate left its end buses unchanged"
    print(f"at start ell = 0 and the bus latents equal stock GridSFM's bit for bit; closing "
          f"line 5's gate moves its end buses by {float(moved[u]):.2f}, {float(moved[v]):.2f}")


def check_long_run_spot() -> None:
    """A live re-score of 2 hold-out scenarios of each 120k-step run equals its record."""
    from powerform.training import case500
    if not (CORPUS_DIR / "case500_goc" / "raw").is_dir():
        raise Skip("no Table 1 corpus; see powerform/data/corpus/README.md, the Table 1 corpus")
    for arch, stem in STEMS.items():
        ck = CHECKPOINTS / f"{stem}_steps120000_seed0.pt"
        if not ck.exists():
            raise Skip(f"no checkpoints; {FETCH_LONG}")
        if arch == "gridsfm" and not (DATAKIT_HOME / "holdout_opfdata").is_dir():
            raise Skip("no .pyg.json export; see powerform/data/corpus/README.md, the Table 1 corpus")
        args = case500.parse_args(["--eval-only", str(ck)])
        live = case500.rescore(ck, args, torch.device("cpu"), limit=2)["per_scenario_cost_mape"]
        want = load(LONG / f"{stem}_steps120000_seed0.json")["per_scenario_cost_mape"][:2]
        other = load(LONG / f"{stem}_steps60000_seed0.json")["per_scenario_cost_mape"][:2]
        if arch == "physics":       # float64 throughout: relative agreement
            diff = lambda ref: max(abs(a - b) / b for a, b in zip(live, ref))  # noqa: E731
            tol, unit = TOL_F64, "relative"
        else:                       # float32, recorded on the GPU: absolute, in percentage points;
            diff = lambda ref: max(abs(a - b) for a, b in zip(live, ref))      # noqa: E731
            tol, unit = TOL_PP, "percentage points"   # a near-zero scenario makes relative meaningless
        assert diff(want) <= tol, f"{arch}: {diff(want):.2e} {unit} > {tol}"
        assert diff(other) > tol, f"{arch}: the 60k record passed"
        print(f"    {arch}: difference {diff(want):.2e} {unit}")


def check_gridsfm_rescore() -> None:
    """The scratch seed-0 model's final evaluation regenerates on the CPU. The pretrained
    models are not re-scored: their line scores collapsed to a spread of about 1e-8, so
    their top-3 choice depends on floating-point rounding and need not repeat."""
    need_case57()
    name = "reconfig_gridsfm_scratch_seed0.pt"
    if not (CHECKPOINTS / name).exists():
        raise Skip(f"{name} absent; fetch with `{FETCH_RECONFIG}`")
    from powerform.eval.scorer import score
    from powerform.experiments.reference import N_SCENARIOS
    from powerform.model.baselines.gridsfm_track import build
    pool_dev, pool_cpu, _ = gridsfm_pool(N_SCENARIOS["val"])
    net = build("scratch", 32)
    net.load_state_dict(torch.load(CHECKPOINTS / name, map_location="cpu", weights_only=True)["model"])
    net.eval()
    run = load(REC / "gridsfm_scratch_seed0.json")
    with torch.no_grad():
        got = score(net, pool_dev, pool_cpu, load(REC / "reference_val.json"), run["k_open"])
    want = {r["metric"]: r["value"] for r in run["rows"] if r["step"] == run["steps"]}
    assert got["infeasible_fraction"] == want["infeasible_fraction"], (got, want)
    assert abs(got["gap_pct_median"] - want["gap_pct_median"]) <= 1e-9, (got, want)
    print(f"scratch seed 0 at step {run['steps']}: infeasible {got['infeasible_fraction']:.3f}, "
          f"median gap {got['gap_pct_median']:+.3f}%, record agrees")


CHECKS = (check_long_run_records, check_long_run_records_export, check_training_curve, check_settled_table, check_figure3,
          check_gridsfm_settings, check_gridsfm_attachment, check_gridsfm_identity_and_gate,
          check_long_run_spot, check_gridsfm_rescore)


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
