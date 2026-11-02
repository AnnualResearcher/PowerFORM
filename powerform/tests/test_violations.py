"""The violation audit: its counts, the printed Table 1 column, and re-scored rows.

The audit's counting rule is checked on a synthetic grid and the column from the
shipped record. With data and checkpoints present, the labels must audit clean and
states must respond to a moved weight; `--deep` re-audits a whole hold-out row,
about 15 minutes per checkpoint on one thread.

`--summarize` compares the `violations.json` that `eval/violations.py` wrote to a
directory, by default outputs/table1, with the record and the printed column. It
exits 0 when every PowerForm row was run and agrees, 1 on any disagreement, 2 when
nothing disagrees but some rows were not run.

    python powerform/tests/test_violations.py [--deep]
    python powerform/tests/test_violations.py --summarize [DIR]
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from powerform.data.corpus.datakit_export import CORPUS_DIR  # noqa: E402
from powerform.data.datakit_loader import HOLDOUT_START  # noqa: E402
from powerform.eval import violations as V  # noqa: E402
from powerform.eval.checkpoint import load_model  # noqa: E402
from powerform.paths import CHECKPOINTS, OUTPUTS, RESULTS  # noqa: E402
from powerform.physics.newton_solver import TransmissionGrid, build_ybus  # noqa: E402

RECORD = RESULTS / "table1" / "violations.json"
RAW = CORPUS_DIR / "case500_goc" / "raw"
FETCH_DATA = "see powerform/data/corpus/README.md, the Table 1 corpus"
FETCH_CKPT = "python scripts/download_checkpoints.py --group table1"
# Table 1, viol./scen., as printed: mean and sd.
PRINTED = {"gnn": ("23.16", "1.34"), "mlp": ("23.64", "1.15"),
           "graphkit_state": ("34.41", "5.4"), "graphkit_decision": ("30.16", ".55"),
           "gridsfm_state": ("24.59", "2.50"), "gridsfm_decision": ("23.99", ".50"),
           "generic": ("23.19", ".74"), "physics": ("23.28", ".55")}
# The rows `eval/violations.py` can re-score; the others ship as records only.
RESCORED = ("physics", "generic")
REL_TOL, ABS_FLOOR = 1e-10, 1e-12


class Skip(Exception):
    pass


def small_grid() -> tuple:
    """Three buses on a triangle, voltages flat, with limits and ratings that a flat
    state just satisfies."""
    n = 3
    g = TransmissionGrid(
        f_bus=torch.tensor([0, 1, 2]), t_bus=torch.tensor([1, 2, 0]),
        br_r=torch.full((3,), 0.01, dtype=torch.float64),
        br_x=torch.full((3,), 0.1, dtype=torch.float64),
        br_b=torch.zeros(3, dtype=torch.float64), tap=torch.ones(3, dtype=torch.float64),
        shift_rad=torch.zeros(3, dtype=torch.float64), gs=torch.zeros(n, dtype=torch.float64),
        bs=torch.zeros(n, dtype=torch.float64), p_load=torch.zeros(n, dtype=torch.float64),
        q_load=torch.zeros(n, dtype=torch.float64),
        is_slack=torch.tensor([True, False, False]))
    st = {"pd": np.zeros(n), "qd": np.zeros(n), "vmin": np.full(n, 0.94), "vmax": np.full(n, 1.06),
          "free": np.array([True, True, False]), "qmin": np.full(n, -1.0), "qmax": np.full(n, 1.0),
          "pmin": np.full(n, -1.0), "pmax": np.full(n, 1.0)}
    return g, build_ybus(g).numpy(), np.full(3, 1.0), st


def check_counting_rule() -> None:
    g, Y, rate, st = small_grid()
    vm, va = np.ones(3), np.zeros(3)
    base = V.audit_state(vm, va, st, g, Y, rate)
    assert (base["v_n"], base["q_n"], base["p_n"], base["n_over"]) == (0, 0, 0, 0), base
    over = vm.copy()
    over[2] = 1.06 + 2e-6
    assert V.audit_state(over, va, st, g, Y, rate)["v_n"] == 1
    within = vm.copy()
    within[2] = 1.06 + 5e-7
    assert V.audit_state(within, va, st, g, Y, rate)["v_n"] == 0
    # slack injection pushed above its box: counted on the slack
    tight = {**st, "pmax": np.full(3, -0.5)}
    rec = V.audit_state(vm, va, tight, g, Y, rate)
    assert rec["p_n_slack"] == 1 and rec["p_n"] == 2 and rec["p_n_nonslack"] == 1, rec
    # a state with an angle spread and a low rating overloads
    rec = V.audit_state(vm, np.array([0.0, 0.2, 0.0]), st, g, Y, np.full(3, 0.5))
    assert rec["n_over"] > 0
    row = V.aggregate([base, V.audit_state(over, va, st, g, Y, rate)])
    assert row["voltage_box"]["per_scenario_mean"] == 0.5
    assert row["violated_inequalities_per_scenario"] == 0.5 and row["scenarios_with_any"] == 1
    print("the tolerance, the slack indicator, thermal and the means count as defined")


def check_printed_column() -> None:
    rows = json.loads(RECORD.read_text())["rows"]
    for row, (mean, sd) in PRINTED.items():
        vals = [rows[f"{row}_seed{s}"]["violated_inequalities_per_scenario"] for s in range(3)]
        got = (f"{np.mean(vals):.2f}", f"{np.std(vals, ddof=1):.{len(sd.split('.')[1])}f}")
        assert got[0] == mean and got[1].lstrip("0") == sd.lstrip("0"), (row, got, mean, sd)
    moved = [rows[f"physics_seed{s}"]["violated_inequalities_per_scenario"] for s in range(3)]
    moved[0] += 0.1
    assert f"{np.mean(moved):.2f}" != PRINTED["physics"][0]
    print(f"{len(PRINTED)} printed rows; a moved value changes the mean")


def need_corpus() -> None:
    if not (RAW / "bus_data.parquet").exists():
        raise Skip(f"case500_goc corpus absent; {FETCH_DATA}")


def need_ckpt(name: str) -> Path:
    ck = CHECKPOINTS / name
    if not ck.exists():
        raise Skip(f"{name} absent; fetch with `{FETCH_CKPT}`")
    return ck


def check_labels_audit_clean() -> None:
    need_corpus()
    grid, Y, rate, n = V.build_grid(RAW)
    statics = V.scenario_statics(RAW, n, grid.is_slack.numpy())
    states = [(s, st["vm_label"], st["va_label"]) for s, st in sorted(statics.items())]
    row = V.audit_states(states, statics, grid, Y, rate)
    assert row["violated_inequalities_per_scenario"] == 0.0, row
    assert int(statics[HOLDOUT_START]["free"].sum()) == json.loads(RECORD.read_text())["n_free_buses"]
    try:
        V.audit_states(states[:3] + states[:1], statics, grid, Y, rate, expect_n=None)
    except SystemExit:
        pass
    else:
        raise AssertionError("a duplicated scenario went through")
    try:
        V.audit_states(states[:3], statics, grid, Y, rate)
    except SystemExit:
        pass
    else:
        raise AssertionError("a short row went through")
    print("the IPOPT labels violate nothing over 500 scenarios; bad scenario sets stop")


def check_states_respond_to_weights() -> None:
    need_corpus()
    model = load_model(need_ckpt("case500_physics_seed0.pt"), "cpu")
    data = V.load_holdout()
    base = list(V.powerform_states(model, data, "cpu", limit=2))
    with torch.no_grad():
        next(iter(model.parameters())).view(-1)[0] += 1e-9
    moved = list(V.powerform_states(model, data, "cpu", limit=2))
    assert [s[0] for s in base] == [HOLDOUT_START, HOLDOUT_START + 1]
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(base, moved))
    print("realised states change with a weight + 1e-9")


def within(a, b) -> bool:
    if isinstance(a, dict):
        return set(a) == set(b) and all(within(a[k], b[k]) for k in a)
    if isinstance(a, float):
        gap = abs(a - b)
        return gap <= ABS_FLOOR or gap <= REL_TOL * max(abs(a), abs(b))
    return a == b


def check_full_row_rescores() -> None:
    if "--deep" not in sys.argv:
        raise Skip("takes about 15 minutes; pass --deep")
    need_corpus()
    ck = need_ckpt("case500_physics_seed0.pt")
    grid, Y, rate, n = V.build_grid(RAW)
    statics = V.scenario_statics(RAW, n, grid.is_slack.numpy())
    data = V.load_holdout()
    V.check_alignment(data, statics)
    row = V.audit_states(V.powerform_states(load_model(ck, "cpu"), data, "cpu"),
                         statics, grid, Y, rate)
    want = json.loads(RECORD.read_text())["rows"]["physics_seed0"]
    assert within(row, want), (row, want)
    print(f"physics seed 0: {row['violated_inequalities_per_scenario']} viol./scen., record agrees")


def compare_row(got: dict | None, want: dict) -> str:
    """'exact', 'within tolerance', 'disagree', or 'not run' for a missing row or a
    subset run. A count is a mean over the same scenarios, so it must match exactly."""
    if got is None or got.get("n_scenarios") != want["n_scenarios"]:
        return "not run"
    if got == want:
        return "exact"
    same_cell = (got["violated_inequalities_per_scenario"]
                 == want["violated_inequalities_per_scenario"])
    return "within tolerance" if same_cell and within(got, want) else "disagree"


def summarize(out_dir: Path) -> int:
    """The re-scorable rows from `out_dir`/violations.json; returns the exit code."""
    if not out_dir.is_dir():
        print(f"{out_dir} does not exist: nothing to compare")
        return 2
    path = out_dir / "violations.json"
    got = json.loads(path.read_text())["rows"] if path.exists() else {}
    record = json.loads(RECORD.read_text())["rows"]
    counts = dict.fromkeys(("exact", "within tolerance", "disagree", "not run"), 0)
    bad_mean = False
    for row in RESCORED:
        verdicts = [compare_row(got.get(f"{row}_seed{s}"), record[f"{row}_seed{s}"])
                    for s in range(3)]
        for v in verdicts:
            counts[v] += 1
        tally = ", ".join(f"{verdicts.count(v)} {v}" for v in counts if v in verdicts)
        mean, sd = PRINTED[row]
        label = f"{row:10s}printed {mean} +- {sd:6s}"
        if "not run" in verdicts:
            print(f"  {label}{'not run' if verdicts.count('not run') == 3 else 'incomplete, ' + tally}")
            continue
        vals = [got[f"{row}_seed{s}"]["violated_inequalities_per_scenario"] for s in range(3)]
        shown = (f"{np.mean(vals):.2f}", f"{np.std(vals, ddof=1):.{len(sd.split('.')[1])}f}")
        bad_mean |= shown[0] != mean or shown[1].lstrip("0") != sd.lstrip("0")
        print(f"  {label}re-scored {shown[0]} +- {shown[1]}  {tally}")
    baselines = sorted(set(PRINTED) - set(RESCORED))
    print(f"{sum(counts.values())} rows: " + ", ".join(f"{n} {k}" for k, n in counts.items())
          + f". Records only: {', '.join(baselines)}.")
    if counts["disagree"] or bad_mean:
        return 1
    return 2 if counts["not run"] else 0


def silent_summary(d: Path) -> int:
    """The exit code alone: these checks run the tool on copies of the records."""
    with contextlib.redirect_stdout(io.StringIO()):
        return summarize(d)


def check_summary_exit_codes() -> None:
    record = json.loads(RECORD.read_text())
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows = {f"{r}_seed{s}": record["rows"][f"{r}_seed{s}"] for r in RESCORED for s in range(3)}
        write = lambda d: (tmp / "violations.json").write_text(json.dumps({"rows": d}))  # noqa: E731
        write(rows)
        assert silent_summary(tmp) == 0
        worst = rows["physics_seed0"]["voltage_box"]["worst"]
        rows["physics_seed0"]["voltage_box"]["worst"] = worst * (1 + 1e-13)
        write(rows)
        assert silent_summary(tmp) == 0, "a rounding difference is not a disagreement"
        rows["physics_seed0"]["voltage_box"]["worst"] = worst * (1 + 1e-6)
        write(rows)
        assert silent_summary(tmp) == 1
        rows["physics_seed0"]["voltage_box"]["worst"] = worst
        rows["physics_seed0"]["n_scenarios"] = 5
        write(rows)
        assert silent_summary(tmp) == 2, "a subset row counts as not run"
        del rows["physics_seed0"]
        write(rows)
        assert silent_summary(tmp) == 2
        assert silent_summary(tmp / "absent") == 2
    print("--summarize on copies of the records: exit 0 when complete, 1 on a moved value, "
          "2 on a missing or subset row")


CHECKS = (check_counting_rule, check_printed_column, check_summary_exit_codes,
          check_labels_audit_clean, check_states_respond_to_weights, check_full_row_rescores)


def main() -> None:
    if "--summarize" in sys.argv:
        rest = sys.argv[sys.argv.index("--summarize") + 1:]
        sys.exit(summarize(Path(rest[0]) if rest else OUTPUTS / "table1"))
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
